"""B1: EEGNet (Lawhern et al. 2018) and a small conv-transformer with switchable tokenisation, written from building blocks."""
import torch
from torch import nn

FS = 160


class TemporalConv(nn.Module):
    """K temporal filters shared by every electrode: (B, C, T) -> (B, K, C, T). Same maths as Conv2d(1, K, (1, L), 'same');
    written as conv1d over (B*C, 1, T) because the 1-input-channel Conv2d is ~3.5x slower on CPU (own timing)."""

    def __init__(self, K, L):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(K, 1, L))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)   # Conv2d's default initialisation

    def forward(self, x):
        B, C, T = x.shape
        z = nn.functional.conv1d(x.reshape(B * C, 1, T), self.weight, padding="same")
        return z.reshape(B, C, -1, T).transpose(1, 2)


class EEGNet(nn.Module):
    """EEGNet-8,2. The paper's numbers are for 128 Hz; every change for 160 Hz is listed in DECISIONS D20.
    kern_len = FS/2 keeps the first filter at 0.5 s (paper: 64 samples at 128 Hz) so its lowest resolvable frequency stays 2 Hz.
    sep_len = 20 keeps the separable filter at 0.5 s after the /4 pool (paper: 16 samples at 32 Hz)."""

    def __init__(self, n_ch=64, n_t=640, n_cls=2, F1=8, D=2, F2=16, kern_len=FS // 2, sep_len=20, drop=0.25):
        super().__init__()
        self.block1 = nn.Sequential(
            TemporalConv(F1, kern_len),
            nn.BatchNorm2d(F1),
            # depthwise spatial filter: one set of D weights over all electrodes per temporal filter (the CSP-like layer)
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop))
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, sep_len), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop))
        self.fc = nn.Linear(F2 * (n_t // 32), n_cls)

    @property
    def spatial(self):
        return self.block1[2]

    def forward(self, x):                 # x: (B, C, T)
        z = self.block2(self.block1(x))
        return self.fc(z.flatten(1))

    def apply_max_norm(self):
        # paper: max-norm 1 on the depthwise spatial kernels, 0.25 on the classifier; keeps any electrode weight from exploding
        with torch.no_grad():
            for w, c in ((self.spatial.weight, 1.0), (self.fc.weight, 0.25)):
                w.copy_(torch.renorm(w, 2, 0, c))


class EEGTransformer(nn.Module):
    """Shared temporal filter bank -> tokens -> 2-layer pre-norm transformer -> CLS -> linear.
    tokens="time":    token t = all electrodes in segment t (depthwise spatial filter, then pooled to 0.2 s segments).
    tokens="channel": token c = one electrode's filtered time series (pooled to 0.5 s segments), no spatial mixing before attention.
    ch_id: add a learned electrode-identity embedding to channel tokens. Without it the encoder sees an unordered set of electrodes.
    Sizes are chosen to stay under 3x EEGNet's parameters (DECISIONS D21)."""

    def __init__(self, tokens="time", ch_id=True, n_ch=64, n_t=640, n_cls=2, K=8, D=2, d=16, heads=2, ff=32,
                 layers=2, kern_len=FS // 2, drop=0.25):
        super().__init__()
        self.tokens, self.ch_id = tokens, ch_id
        # same first layer as EEGNet so the token axis is the only difference to compare, not the frequency front end
        self.temporal = nn.Sequential(TemporalConv(K, kern_len), nn.BatchNorm2d(K))
        if tokens == "time":
            self.spatial = nn.Conv2d(K, K * D, (n_ch, 1), groups=K, bias=False)
            self.tok = nn.Sequential(nn.BatchNorm2d(K * D), nn.ELU(), nn.AvgPool2d((1, 32)), nn.Dropout(drop))
            n_tok, d_in = n_t // 32, K * D
            self.pos = nn.Parameter(torch.zeros(1, n_tok, d))   # time position; order of segments matters for ERD timing
        else:
            self.tok = nn.Sequential(nn.ELU(), nn.AvgPool2d((1, 80)), nn.Dropout(drop))
            n_tok, d_in = n_ch, K * (n_t // 80)
            self.pos = nn.Parameter(torch.zeros(1, n_ch, d)) if ch_id else None
        self.proj = nn.Linear(d_in, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, drop, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)
        for p in (self.pos, self.cls):
            if p is not None:
                nn.init.normal_(p, std=0.02)

    def embed(self, x):
        """(B, C, T) -> token embeddings (B, N, d) before the CLS token and position/identity are added."""
        z = self.temporal(x)                                  # (B, K, C, T)
        if self.tokens == "time":
            z = self.tok(self.spatial(z))[:, :, 0]            # (B, K*D, n_tok)
            z = z.transpose(1, 2)
        else:
            z = self.tok(z)                                   # (B, K, C, n_seg)
            z = z.permute(0, 2, 1, 3).flatten(2)              # (B, C, K*n_seg)
        return self.proj(z)

    def forward(self, x, token_mask=None, return_tokens=False):
        """token_mask (B, N) bool: True = delete this token (used by the faithfulness test, B2)."""
        t = self.embed(x)
        if self.pos is not None:
            t = t + self.pos
        if token_mask is not None:
            t = t.masked_fill(token_mask[..., None], 0.0)
        h = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        mask = None if token_mask is None else torch.cat([torch.zeros_like(token_mask[:, :1]), token_mask], 1)
        h = self.encoder(h, src_key_padding_mask=mask)
        out = self.head(self.norm(h[:, 0]))
        return (out, t) if return_tokens else out

    def apply_max_norm(self):
        if self.tokens == "time":
            with torch.no_grad():
                self.spatial.weight.copy_(torch.renorm(self.spatial.weight, 2, 0, 1.0))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


MODELS = {
    "eegnet": lambda **k: EEGNet(**k),
    "tf_time": lambda **k: EEGTransformer("time", **k),
    "tf_chan_id": lambda **k: EEGTransformer("channel", ch_id=True, **k),
    "tf_chan_noid": lambda **k: EEGTransformer("channel", ch_id=False, **k),
}

if __name__ == "__main__":
    base = n_params(EEGNet())
    for name, f in MODELS.items():
        m = f()
        x = torch.randn(3, 64, 640)
        print(f"{name:13s} params={n_params(m):6d} ({n_params(m) / base:.2f}x EEGNet) out={tuple(m(x).shape)}")
