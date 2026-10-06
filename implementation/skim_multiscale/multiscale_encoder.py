import math
from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from espnet2.enh.encoder.abs_encoder import AbsEncoder
except ImportError:
    class AbsEncoder(nn.Module):
        pass


class CumulativeLayerNorm(nn.Module):
    """Cumulative Layer Normalization for causal sequence modeling."""

    def __init__(self, channel_size: int, eps: float = 1e-5):
        super().__init__()
        self.channel_size = channel_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, channel_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, channel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Channel)
        mean = torch.cumsum(x, dim=1) / torch.arange(
            1, x.size(1) + 1, device=x.device, dtype=x.dtype
        ).view(1, -1, 1)
        var = torch.cumsum((x - mean) ** 2, dim=1) / torch.arange(
            1, x.size(1) + 1, device=x.device, dtype=x.dtype
        ).view(1, -1, 1)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.gamma + self.beta


class MultiScaleConvEncoder(AbsEncoder):
    """Multi-Scale 1D Convolutional Encoder for Speech Separation.

    Inspired by SpEx+ (Ge et al., 2020) and TDNext (Ye et al., 2025).
    Extracts multi-resolution acoustic features using multiple 1D convolution
    kernels (e.g., Short, Middle, Long) with a shared uniform stride, then
    fuses them into a unified latent representation.

    Args:
        channel: Feature dimension for each scale branch (default: 256).
        out_channel: Output feature dimension after fusion (default: 256).
        kernel_sizes: Tuple of kernel lengths (default: (16, 32, 64)).
        stride: Stride size shared across all scales (default: 8).
        causal: If True, uses left-padding for streaming/causal low-latency mode.
        nonlinear: Nonlinearity after projection ('relu' or 'prelu').
        norm_type: Normalization type ('gLN' for global, 'cLN' for causal).
    """

    def __init__(
        self,
        channel: int = 256,
        out_channel: Optional[int] = None,
        kernel_sizes: Tuple[int, ...] = (16, 32, 64),
        stride: int = 8,
        causal: bool = False,
        nonlinear: str = "relu",
        norm_type: Optional[str] = None,
    ):
        super().__init__()
        self.channel = channel
        self.out_channel = out_channel if out_channel is not None else channel
        self.kernel_sizes = list(kernel_sizes)
        self.stride = stride
        self.causal = causal

        # Cabang konvolusi 1D paralel untuk setiap skala kernel
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=channel,
                kernel_size=k,
                stride=stride,
                bias=False,
            )
            for k in self.kernel_sizes
        ])

        # Proyeksi 1x1 untuk fusi fitur multi-skala: (channel * num_scales) -> out_channel
        num_scales = len(self.kernel_sizes)
        self.proj = nn.Conv1d(channel * num_scales, self.out_channel, kernel_size=1)

        # Normalisasi
        if norm_type is None:
            norm_type = "cLN" if causal else "gLN"
        self.norm_type = norm_type

        if self.norm_type == "cLN":
            self.norm = CumulativeLayerNorm(self.out_channel)
        else:
            self.norm = nn.GroupNorm(1, self.out_channel, eps=1e-5)

        # Aktivasi non-linear
        if nonlinear == "relu":
            self.act = nn.ReLU()
        elif nonlinear == "prelu":
            self.act = nn.PReLU()
        else:
            raise ValueError(f"Unsupported nonlinear: {nonlinear}")

    def _pad_input(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Menghitung dan menerapkan padding agar dimensi temporal konsisten."""
        pad_total = kernel_size - self.stride
        if pad_total <= 0:
            return x

        if self.causal:
            # Mode kausal: hanya padding di awal (masa lalu)
            return F.pad(x, (pad_total, 0))
        else:
            # Mode non-kausal: padding simetris kiri dan kanan
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return F.pad(x, (pad_left, pad_right))

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass multi-scale encoder.

        Args:
            input: Time-domain audio (Batch, Samples).
            ilens: Sequence lengths tensor (Batch,).

        Returns:
            feature: Laten feature tensor (Batch, Frames, out_channel).
            flens: Frame lengths tensor (Batch,).
        """
        assert input.dim() == 2, "Input audio must be 2D: (Batch, Samples)"
        x = input.unsqueeze(1)  # (Batch, 1, Samples)

        # 1. Ekstraksi fitur konvolusi untuk tiap skala kernel
        branch_feats = []
        for conv, k in zip(self.conv_branches, self.kernel_sizes):
            x_padded = self._pad_input(x, k)
            feat = self.act(conv(x_padded))  # (Batch, channel, T_k)
            branch_feats.append(feat)

        # 2. Penyelarasan panjang temporal (antisipasi pembulatan integer padding)
        min_frames = min(f.size(2) for f in branch_feats)
        aligned_feats = [f[:, :, :min_frames] for f in branch_feats]

        # 3. Konkatenasi kanal antar-skala (Batch, channel * num_scales, min_frames)
        fused = torch.cat(aligned_feats, dim=1)

        # 4. Proyeksi 1x1 dan Normalisasi
        projected = self.proj(fused)  # (Batch, out_channel, min_frames)
        if self.norm_type == "cLN":
            # cLN menerima (Batch, Frames, Channel)
            projected = projected.transpose(1, 2)
            normalized = self.norm(projected)  # (Batch, Frames, out_channel)
            feature = normalized
        else:
            # GroupNorm menerima (Batch, Channel, Frames)
            normalized = self.norm(projected)
            feature = normalized.transpose(1, 2).contiguous()  # (Batch, Frames, out_channel)

        flens = torch.full_like(ilens, min_frames)
        return feature, flens

    @property
    def output_dim(self) -> int:
        return self.out_channel
