import math
import torch
from torch import nn
from torch.nn import functional as F

from openvoice import commons
from openvoice import modules
from openvoice import attentions

from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from openvoice.commons import init_weights, get_padding, sequence_mask


class TextEncoder(nn.Module):
	def __init__(self,
			n_vocab,
			out_channels,
			hidden_channels,
			filter_channels,
			n_heads,
			n_layers,
			kernel_size,
			p_dropout):
		super().__init__()
		self.n_vocab = n_vocab
		self.out_channels = out_channels
		self.hidden_channels = hidden_channels
		self.filter_channels = filter_channels
		self.n_heads = n_heads
		self.n_layers = n_layers
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout

		self.emb = nn.Embedding(n_vocab, hidden_channels)
		nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

		self.encoder = attentions.Encoder(
			hidden_channels,
			filter_channels,
			n_heads,
			n_layers,
			kernel_size,
			p_dropout)
		self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

	def forward(self, x, x_lengths):
		x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
		x = torch.transpose(x, 1, -1) # [b, h, t]
		x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

		x = self.encoder(x * x_mask, x_mask)
		stats = self.proj(x) * x_mask

		m, logs = torch.split(stats, self.out_channels, dim=1)
		return x, m, logs, x_mask
     

class DurationPredictor(nn.Module):
    def __init__(
        self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
               
class StochasticDurationPredictor(nn.Module):
	def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
		super().__init__()
		filter_channels = in_channels # it needs to be removed from future version.
		self.in_channels = in_channels
		self.filter_channels = filter_channels
		self.kernel_size = kernel_size
		self.p_dropout = p_dropout
		self.n_flows = n_flows
		self.gin_channels = gin_channels

		self.log_flow = modules.Log()
		self.flows = nn.ModuleList()
		self.flows.append(modules.ElementwiseAffine(2))
		for i in range(n_flows):
			self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.flows.append(modules.Flip())

		self.post_pre = nn.Conv1d(1, filter_channels, 1)
		self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
		self.post_flows = nn.ModuleList()
		self.post_flows.append(modules.ElementwiseAffine(2))
		for i in range(4):
			self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
			self.post_flows.append(modules.Flip())

		self.pre = nn.Conv1d(in_channels, filter_channels, 1)
		self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
		self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
		if gin_channels != 0:
			self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

	def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
		x = torch.detach(x)
		x = self.pre(x)
		if g is not None:
			g = torch.detach(g)
			x = x + self.cond(g)
		x = self.convs(x, x_mask)
		x = self.proj(x) * x_mask

		if not reverse:
			flows = self.flows
			assert w is not None

			logdet_tot_q = 0
			h_w = self.post_pre(w)
			h_w = self.post_convs(h_w, x_mask)
			h_w = self.post_proj(h_w) * x_mask
			e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
			z_q = e_q
			for flow in self.post_flows:
				z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
				logdet_tot_q += logdet_q
			z_u, z1 = torch.split(z_q, [1, 1], 1)
			u = torch.sigmoid(z_u) * x_mask
			z0 = (w - u) * x_mask
			logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
			logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

			logdet_tot = 0
			z0, logdet = self.log_flow(z0, x_mask)
			logdet_tot += logdet
			z = torch.cat([z0, z1], 1)
			for flow in flows:
				z, logdet = flow(z, x_mask, g=x, reverse=reverse)
				logdet_tot = logdet_tot + logdet
			nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
			return nll + logq # [b]
		else:
			flows = list(reversed(self.flows))
			flows = flows[:-2] + [flows[-1]] # remove a useless vflow
			z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
			for flow in flows:
				z = flow(z, x_mask, g=x, reverse=reverse)
			z0, z1 = torch.split(z, [1, 1], 1)
			logw = z0
			return logw

class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None, tau=1.0):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * tau * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()


class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, spec_channels, gin_channels=0, layernorm=True):
        super().__init__()
        self.spec_channels = spec_channels
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters
        convs = [
            weight_norm(
                nn.Conv2d(
                    in_channels=filters[i],
                    out_channels=filters[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=(1, 1),
                )
            )
            for i in range(K)
        ]
        self.convs = nn.ModuleList(convs)

        out_channels = self.calculate_channels(spec_channels, 3, 2, 1, K)
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels,
            hidden_size=256 // 2,
            batch_first=True,
        )
        self.proj = nn.Linear(128, gin_channels)
        if layernorm:
            self.layernorm = nn.LayerNorm(self.spec_channels)
        else:
            self.layernorm = None

    def forward(self, inputs, mask=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # out --- [1, N, 128]

        return self.proj(out.squeeze(0))

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

    def infer(self, inputs, lengths=None):
        N = inputs.size(0)

        out = inputs.view(N, 1, -1, self.spec_channels)  # [N, 1, Ty, n_freqs]
        if self.layernorm is not None:
            out = self.layernorm(out)

        if lengths is not None:
            out = out * sequence_mask(lengths, out.shape[2]).view(N, 1, -1, 1)

        for conv in self.convs:
            out = conv(out)
            # out = wn(out)
            out = F.relu(out)  # [N, 128, Ty//2^K, n_mels//2^K]
            
            if lengths is not None:
                lengths = (lengths - 3 + 2 * 1) // 2 + 1
                out = out * sequence_mask(lengths, out.shape[2]).view(N, 1, -1, 1)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        T = out.size(1)
        N = out.size(0)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        self.gru.flatten_parameters()
        memory, out = self.gru(out)  # memory [N, T, 128], out [1, N, 128], memory[:,-1,:] is out[0]

        if lengths is not None:
            out = torch.index_select(
                memory.reshape(N * T, -1),
                0,
                torch.arange(N) * T + lengths - 1
            ).view(1, N, -1)

        return self.proj(out.squeeze(0)) 


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
            channels,
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            n_flows=4,
            gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        zero_g=False,
        **kwargs
    ):
        super().__init__()

        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        self.n_speakers = n_speakers
        if n_speakers == 0:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels)
        else:
            self.enc_p = TextEncoder(n_vocab,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout)
            self.sdp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        self.zero_g = zero_g

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., sdp_ratio=0.2, max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1) # [b, h, 1]
        else:
            g = None

        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * sdp_ratio \
            + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)

        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:,:,:max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, tau=1.0):
        g_src = sid_src
        g_tgt = sid_tgt
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src if not self.zero_g else torch.zeros_like(g_src), tau=tau)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt if not self.zero_g else torch.zeros_like(g_tgt))
        return o_hat, y_mask, (z, z_p, z_hat)
