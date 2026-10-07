from typing import Tuple, Optional
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
    kernels (e.g., Short, Middle, Long). Retains separate scale channels when
    preserve_scales=True; legacy mode fuses them. Independent strides resample
    native branch features onto the finest frame grid in noncausal mode.

    Args:
        channel: Feature dimension for each scale branch (default: 256).
        out_channel: Output feature dimension in legacy fusion mode.
        kernel_sizes: Tuple of kernel lengths (default: (16, 32, 64)).
        stride: Shared stride or a tuple containing one stride per scale.
        causal: If True, uses left-padding for streaming/causal low-latency mode.
        nonlinear: Nonlinearity on each convolution branch ('relu' or 'prelu').
        norm_type: Legacy fusion normalization ('gLN' or cumulative 'cLN').
        preserve_scales: Return concatenated, unprojected scale features for
            scale-specific masking/decoding. False retains legacy checkpoints.
    """

    def __init__(
        self,
        channel: int = 256,
        out_channel: Optional[int] = None,
        kernel_sizes: Tuple[int, ...] = (16, 32, 64),
        stride: int | Tuple[int, ...] = 8,
        causal: bool = False,
        nonlinear: str = "relu",
        norm_type: Optional[str] = None,
        preserve_scales: bool = False,
    ):
        super().__init__()
        self.channel = channel
        self.out_channel = out_channel if out_channel is not None else channel
        self.kernel_sizes = list(kernel_sizes)
        self.strides = (stride,) * len(self.kernel_sizes) if isinstance(stride, int) else tuple(stride)
        self.stride = min(self.strides) if self.strides else 0
        self.causal = causal
        self.preserve_scales = preserve_scales
        if (not self.kernel_sizes or len(self.strides) != len(self.kernel_sizes)
                or any(s <= 0 or s > k for k, s in zip(self.kernel_sizes, self.strides))):
            raise ValueError("Provide one stride per kernel, each satisfying 0 < stride <= kernel")
        if len(set(self.strides)) > 1 and (not preserve_scales or causal):
            raise ValueError("Independent strides require preserve_scales=True and causal=False")

        # Cabang konvolusi 1D paralel untuk setiap skala kernel
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=1,
                out_channels=channel,
                kernel_size=k,
                stride=s,
                bias=False,
            )
            for k, s in zip(self.kernel_sizes, self.strides)
        ])

        # Proyeksi 1x1 untuk fusi fitur multi-skala: (channel * num_scales) -> out_channel
        num_scales = len(self.kernel_sizes)
        if not preserve_scales:
            self.proj = nn.Conv1d(channel * num_scales, self.out_channel, kernel_size=1)

        # Normalisasi
        if norm_type is None:
            norm_type = "cLN" if causal else "gLN"
        self.norm_type = norm_type

        if not preserve_scales:
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

    def _pad_input(self, x: torch.Tensor, kernel_size: int, stride: int) -> torch.Tensor:
        """Menghitung dan menerapkan padding agar dimensi temporal konsisten."""
        pad_total = kernel_size - stride
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
            feature: (Batch, Frames, output_dim); concatenated scale channels
                when preserve_scales=True.
            flens: Frame lengths tensor (Batch,).
        """
        assert input.dim() == 2, "Input audio must be 2D: (Batch, Samples)"
        x = input.unsqueeze(1)  # (Batch, 1, Samples)

        # 1. Ekstraksi fitur konvolusi untuk tiap skala kernel
        branch_feats = []
        for conv, k, s in zip(self.conv_branches, self.kernel_sizes, self.strides):
            # Every branch keeps its partial final frame at its native stride.
            branch_input = F.pad(x, (0, (-input.size(1)) % s)) if self.preserve_scales else x
            x_padded = self._pad_input(branch_input, k, s)
            feat = self.act(conv(x_padded))  # (Batch, channel, T_k)
            branch_feats.append(feat)

        # 2. Penyelarasan panjang temporal (antisipasi pembulatan integer padding)
        if self.preserve_scales:
            frames = (input.size(1) + self.stride - 1) // self.stride
            aligned_feats = [
                F.interpolate(f, size=frames, mode="linear", align_corners=False)
                if f.size(2) != frames else f for f in branch_feats
            ]
        else:
            frames = min(f.size(2) for f in branch_feats)
            aligned_feats = [f[:, :, :frames] for f in branch_feats]

        # 3. Konkatenasi kanal antar-skala (Batch, channel * num_scales, min_frames)
        fused = torch.cat(aligned_feats, dim=1)
        if self.preserve_scales:
            flens = (ilens + self.stride - 1) // self.stride
            return fused.transpose(1, 2).contiguous(), flens

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

        flens = ilens // self.stride
        return feature, flens

    @property
    def output_dim(self) -> int:
        if self.preserve_scales:
            return self.channel * len(self.kernel_sizes)
        return self.out_channel
