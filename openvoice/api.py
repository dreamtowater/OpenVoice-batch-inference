import torch
import numpy as np
import re
import soundfile
from openvoice import utils
from openvoice import commons
import os
import librosa
from openvoice.text import text_to_sequence
from openvoice.mel_processing import spectrogram_torch
from openvoice.models import SynthesizerTrn
from functools import lru_cache
from torch.nn.utils.rnn import pad_sequence


class OpenVoiceBaseClass(object):
    def __init__(self, 
                config_path, 
                device='cuda:0'):
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = utils.get_hparams_from_file(config_path)

        model = SynthesizerTrn(
            len(getattr(hps, 'symbols', [])),
            hps.data.filter_length // 2 + 1,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.hps = hps
        self.device = device

    def load_ckpt(self, ckpt_path):
        checkpoint_dict = torch.load(ckpt_path, map_location=torch.device(self.device))
        a, b = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        print("Loaded checkpoint '{}'".format(ckpt_path))
        print('missing/unexpected keys:', a, b)


class BaseSpeakerTTS(OpenVoiceBaseClass):
    language_marks = {
        "english": "EN",
        "chinese": "ZH",
    }

    @staticmethod
    def get_text(text, hps, is_symbol):
        text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05)/speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language_str):
        texts = utils.split_sentence(text, language_str=language_str)
        print(" > Text splitted to sentences.")
        print('\n'.join(texts))
        print(" > ===========================")
        return texts

    def tts(self, text, output_path, speaker, language='English', speed=1.0):
        mark = self.language_marks.get(language.lower(), None)
        assert mark is not None, f"language {language} is not supported"

        texts = self.split_sentences_into_pieces(text, mark)

        audio_list = []
        for t in texts:
            t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            t = f'[{mark}]{t}[{mark}]'
            stn_tst = self.get_text(t, self.hps, False)
            device = self.device
            speaker_id = self.hps.speakers[speaker]
            with torch.no_grad():
                x_tst = stn_tst.unsqueeze(0).to(device)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                sid = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=0.667, noise_scale_w=0.6,
                                    length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
            audio_list.append(audio)
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            soundfile.write(output_path, audio, self.hps.data.sampling_rate)


class ToneColorConverter(OpenVoiceBaseClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if kwargs.get('enable_watermark', True):
            import wavmark
            self.watermark_model = wavmark.load_model().to(self.device)
        else:
            self.watermark_model = None
        self.version = getattr(self.hps, '_version_', "v1")



    def extract_se(self, ref_wav_list, se_save_path=None):
        if isinstance(ref_wav_list, str):
            ref_wav_list = [ref_wav_list]
        
        device = self.device
        hps = self.hps
        gs = []
        
        for fname in ref_wav_list:
            audio_ref, sr = librosa.load(fname, sr=hps.data.sampling_rate)
            y = torch.FloatTensor(audio_ref)
            y = y.to(device)
            y = y.unsqueeze(0)
            y = spectrogram_torch(y, hps.data.filter_length,
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                        center=False).to(device)
            with torch.no_grad():
                g = self.model.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
                gs.append(g.detach())
        gs = torch.stack(gs).mean(0)

        if se_save_path is not None:
            os.makedirs(os.path.dirname(se_save_path), exist_ok=True)
            torch.save(gs.cpu(), se_save_path)

        return gs

    def convert(self, audio_src_path, src_se, tgt_se, output_path=None, tau=0.3, message="default"):
        hps = self.hps
        # load audio
        audio, sample_rate = librosa.load(audio_src_path, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio).float()
        
        with torch.no_grad():
            y = torch.FloatTensor(audio).to(self.device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    center=False).to(self.device)
            spec_lengths = torch.LongTensor([spec.size(-1)]).to(self.device)
            audio = self.model.voice_conversion(spec, spec_lengths, sid_src=src_se, sid_tgt=tgt_se, tau=tau)[0][
                        0, 0].data.cpu().float().numpy()
            audio = self.add_watermark(audio, message)
            if output_path is None:
                return audio
            else:
                soundfile.write(output_path, audio, hps.data.sampling_rate)
    
    def add_watermark(self, audio, message):
        if self.watermark_model is None:
            return audio
        device = self.device
        bits = utils.string_to_bits(message).reshape(-1)
        n_repeat = len(bits) // 32

        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to add watermark')
                break
            message_npy = bits[n * 32: (n + 1) * 32]
            
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(device)[None]
                message_tensor = torch.FloatTensor(message_npy).to(device)[None]
                signal_wmd_tensor = self.watermark_model.encode(signal, message_tensor)
                signal_wmd_npy = signal_wmd_tensor.detach().cpu().squeeze()
            audio[(coeff * n) * K: (coeff * n + 1) * K] = signal_wmd_npy
        return audio

    def detect_watermark(self, audio, n_repeat):
        bits = []
        K = 16000
        coeff = 2
        for n in range(n_repeat):
            trunck = audio[(coeff * n) * K: (coeff * n + 1) * K]
            if len(trunck) != K:
                print('Audio too short, fail to detect watermark')
                return 'Fail'
            with torch.no_grad():
                signal = torch.FloatTensor(trunck).to(self.device).unsqueeze(0)
                message_decoded_npy = (self.watermark_model.decode(signal) >= 0.5).int().detach().cpu().numpy().squeeze()
            bits.append(message_decoded_npy)
        bits = np.stack(bits).reshape(-1, 8)
        message = utils.bits_to_string(bits)
        return message


class ToneColorConverterForBatchInference(ToneColorConverter):
    @lru_cache()
    def load_audio_and_calc_spec(self, audio_path: str):
        hps = self.hps
        audio, sample_rate = librosa.load(audio_path, sr=hps.data.sampling_rate)
        audio = torch.tensor(audio, device=self.device).float()
        
        spectrogram = spectrogram_torch(
            audio.unsqueeze(0), hps.data.filter_length,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            center=False
        ).to(self.device).squeeze(0).transpose(0, 1)

        return audio, spectrogram # audio [T_audio] , spectrogram [n_mels, T_spec]

    @torch.inference_mode()
    def extract_se_batch(self, ref_wav_list:list[str]):
        inputs, lengths = [], []
        for audio_path in ref_wav_list:
            audio, spec = self.load_audio_and_calc_spec(audio_path)

            inputs.append(spec)
            lengths.append(spec.shape[0])

        inputs = pad_sequence(inputs, batch_first=True)
        lengths = torch.tensor(lengths, dtype=torch.int64, device=self.device)

        gs = self.model.ref_enc.infer(inputs, lengths).unsqueeze(-1)
        
        return gs # [N, 256, 1], N=len(ref_wav_list)

    @torch.inference_mode()
    def convert_batch(self, audio_src_paths:list[str], src_ses, tgt_ses, output_paths:list[str]=None, tau=0.3):
        audio_lengths, specs, spec_lengths = [], [], []
        for audio_path in audio_src_paths:
            audio, spec = self.load_audio_and_calc_spec(audio_path)
            audio_lengths.append(audio.shape[0])
            specs.append(spec)
            spec_lengths.append(spec.shape[0])
        
        specs = pad_sequence(specs, True).transpose(1,2)
        spec_lengths = torch.tensor(spec_lengths, dtype=torch.int64, device=self.device)

        converted_audios = self.model.voice_conversion(
            specs, spec_lengths, sid_src=src_ses, sid_tgt=tgt_ses, tau=tau
        )[0].squeeze(1).data.cpu().float().numpy()

        if output_paths is not None:
            for converted_audio, audio_length, output_path in zip(converted_audios, audio_lengths, output_paths):
                converted_audio = converted_audio[:audio_length]
                soundfile.write(output_path, converted_audio, self.hps.data.sampling_rate)
        else:
            return [converted_audio[:audio_length] for converted_audio, audio_length in zip(converted_audios, audio_lengths)]
            
