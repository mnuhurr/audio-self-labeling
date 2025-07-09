import torch
import torchaudio


def model_size(model: torch.nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])


@torch.jit.script
def mask_tokens(x: torch.Tensor, mask_token: torch.Tensor, p: float) -> torch.Tensor:
    batch_size, seq_len, _ = x.shape

    mask_pos = torch.rand(batch_size, seq_len, device=x.device) < p
    x[mask_pos] = mask_token.to(x.dtype)

    return x


@torch.jit.script
def conv_layer_norm(x: torch.Tensor,
                    weight: torch.Tensor | None = None,
                    bias: torch.Tensor | None = None,
                    eps: float = 1e-5,
                    dim: int = 1) -> torch.Tensor:

    mean = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdim=True)

    xn = (x - mean) / (std + eps)
    if weight is not None:
        xn = weight * xn

    if bias is not None:
        xn = xn + bias

    return xn


class DropPath(torch.nn.Module):
    """stochastic depth/drop path"""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()

        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, device=x.device).bernoulli_(keep_prob)
        if self.scale_by_keep:
            mask.div_(keep_prob)

        return mask * x


class ConvLayerNorm(torch.nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(1, n_channels, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return conv_layer_norm(x, weight=self.weight, bias=self.bias, eps=self.eps)


class MelScaling(torch.nn.Module):
    def __init__(self, n_mels: int, mu: float = 0.0, sigma: float = 1.0):
        super().__init__()
        log_var = 2 * torch.log(torch.as_tensor(sigma))
        self.mean = torch.nn.Parameter(mu * torch.ones(1, n_mels, 1))
        self.log_var = torch.nn.Parameter(log_var * torch.ones(1, n_mels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * self.log_var)
        return (x - self.mean) / std


class GlobalResponseNormalization(torch.nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, n_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input: (batch, c, h, w)"""
        gx = torch.norm(x, p=2, dim=(-2, -1), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class SpecAugment(torch.nn.Module):
    def __init__(self,
                 n_time_mask: int,
                 n_freq_mask: int,
                 time_mask_param: int = 50,
                 freq_mask_param: int = 16):
        super().__init__()

        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)

        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        for _ in range(self.n_time_mask):
            x = self.time_mask(x)

        for _ in range(self.n_freq_mask):
            x = self.freq_mask(x)

        return x

