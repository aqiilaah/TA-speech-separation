from typing import Tuple, Optional
import torch
import torch.nn as nn

try:
    from espnet2.enh.decoder.abs_decoder import AbsDecoder
except ImportError:
    class AbsDecoder(nn.Module):
        pass


class MultiScaleConvDecoder(AbsDecoder):
    """1D Transposed Convolutional Decoder for Speech Separation.

    Reconstructs time-domain speech waveforms from latent representations.
    Matches the stride of the MultiScaleConvEncoder and pairs with the primary
    resolution kernel (default: 16).

    Args:
        channel: Input feature dimension from separator (default: 256).
        kernel_size: Kernel size of transposed convolution (default: 16).
        stride: Stride of transposed convolution (default: 8).
    """

    def __init__(
        self,
        channel: int = 256,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.channel = channel
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv_trans1d = nn.ConvTranspose1d(
            in_channels=channel,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

    def forward(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for decoder.

        Args:
            input: Latent feature tensor (Batch, Time, Channel).
            ilens: Audio sequence length reference (Batch,).

        Returns:
            wav: Reconstructed waveform (Batch, Samples).
            olens: Output samples length (Batch,).
        """
        # input: (Batch, Time, Channel) -> (Batch, Channel, Time)
        x = input.transpose(1, 2)
        wav = self.conv_trans1d(x)  # (Batch, 1, Samples)
        wav = wav.squeeze(1)        # (Batch, Samples)

        # Truncate / pad to match original audio sample length if ilens provided
        target_len = int(ilens.max().item())
        if wav.size(1) > target_len:
            wav = wav[:, :target_len]
        elif wav.size(1) < target_len:
            wav = nn.functional.pad(wav, (0, target_len - wav.size(1)))

        return wav, ilens
