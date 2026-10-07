from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from espnet2.enh.decoder.abs_decoder import AbsDecoder
except ImportError:
    class AbsDecoder(nn.Module):
        pass


class MultiScaleConvDecoder(AbsDecoder):
    """Decode each masked encoder scale with its matching synthesis kernel.

    ``forward_scales`` returns all waveforms for weighted training losses.
    ``forward`` selects the short waveform, following the authors' TDNext
    decoding script. The scale outputs are not averaged.

    ``channel`` is the channel count per scale. Input is (B, T, channel *
    num_scales), ordered like ``kernel_sizes``. Omitting ``kernel_sizes``
    retains the single-scale decoder and state keys of older checkpoints.
    ``stride`` accepts a shared integer or one stride per scale. Independent
    strides restore each branch's native frame count before decoding.
    """

    def __init__(
        self,
        channel: int = 256,
        kernel_size: Optional[int] = None,
        stride: int | Tuple[int, ...] = 8,
        kernel_sizes: Optional[Tuple[int, ...]] = None,
        causal: bool = False,
    ):
        super().__init__()
        if kernel_size is not None and kernel_sizes is not None:
            raise ValueError("Specify kernel_size (legacy) or kernel_sizes, not both")
        if kernel_size is None and kernel_sizes is None:
            kernel_size = 16
        self.channel = channel
        self.kernel_size = kernel_size
        self.causal = causal
        self.kernel_sizes = tuple(kernel_sizes) if kernel_size is None else (kernel_size,)
        self.strides = (stride,) * len(self.kernel_sizes) if isinstance(stride, int) else tuple(stride)
        self.stride = min(self.strides) if self.strides else 0
        if (not self.kernel_sizes or len(self.strides) != len(self.kernel_sizes)
                or any(s <= 0 or s > k for k, s in zip(self.kernel_sizes, self.strides))):
            raise ValueError("Provide one stride per kernel, each satisfying 0 < stride <= kernel")
        if causal and len(set(self.strides)) > 1:
            raise ValueError("Independent strides require causal=False")
        if tuple(sorted(self.kernel_sizes)) != self.kernel_sizes:
            raise ValueError("kernel_sizes must be ordered from short to long")

        if kernel_size is not None:
            self.conv_trans1d = nn.ConvTranspose1d(channel, 1, kernel_size, stride=self.strides[0], bias=False)
        else:
            self.deconv_branches = nn.ModuleList([
                nn.ConvTranspose1d(channel, 1, k, stride=s, bias=False)
                for k, s in zip(self.kernel_sizes, self.strides)
            ])

    def _decode(self, feature, decoder, kernel_size, stride, ilens):
        target_len = int(ilens.max().item())
        if len(set(self.strides)) > 1:
            native_frames = (target_len + stride - 1) // stride
            if feature.size(1) != native_frames:
                feature = F.interpolate(feature.transpose(1, 2), size=native_frames,
                                        mode="linear", align_corners=False).transpose(1, 2)
        wav = decoder(feature.transpose(1, 2)).squeeze(1)
        if self.kernel_size is None:
            # Undo the corresponding encoder's left/symmetric padding.
            left = kernel_size - stride if self.causal else (kernel_size - stride) // 2
            wav = wav[:, left:]
        wav = wav[:, :target_len]
        wav = F.pad(wav, (0, max(0, target_len - wav.size(1))))
        valid = torch.arange(target_len, device=wav.device)[None, :] < ilens[:, None]
        return wav.masked_fill(~valid, 0)

    def _validate_input(self, input):
        expected_channels = self.channel * len(self.kernel_sizes)
        if input.dim() != 3 or input.size(-1) != expected_channels:
            raise ValueError(f"Expected (B, T, {expected_channels}) latent features")

    def forward_scales(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return one (B, Samples) waveform per scale, fitted to ``ilens``."""
        self._validate_input(input)
        branches = [self.conv_trans1d] if self.kernel_size is not None else self.deconv_branches
        waveforms = [
            self._decode(feature, decoder, k, s, ilens)
            for feature, decoder, k, s in zip(input.split(self.channel, dim=-1), branches, self.kernel_sizes, self.strides)
        ]
        return waveforms, ilens

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the short-scale waveform for inference."""
        self._validate_input(input)
        decoder = self.conv_trans1d if self.kernel_size is not None else self.deconv_branches[0]
        wav = self._decode(input[..., :self.channel], decoder, self.kernel_sizes[0], self.strides[0], ilens)
        return wav, ilens
