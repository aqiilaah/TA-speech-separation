from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

try:
    from espnet2.enh.separator.abs_separator import AbsSeparator
except ImportError:
    class AbsSeparator(nn.Module):
        pass

from implementation.skim_multiscale.skim_multiscale import SkiM


class SkiMMultiScaleSeparator(AbsSeparator):
    """SkiM Separator for Speech Separation with Multi-Scale inputs.

    Accepts fused multi-scale latent features, processes them with Skipping
    Memory LSTM (SegLSTM for intra-segment and MemLSTM for inter-segment),
    and generates estimation masks for each source speaker.

    Args:
        input_dim: Dimension of latent features (from MultiScaleConvEncoder, default: 256).
        causal: Whether the separator is causal (streaming) or non-causal.
        num_spk: Number of speakers to separate (e.g. 2 or 3).
        predict_noise: Whether to output an additional noise mask.
        nonlinear: Activation for masks ('relu', 'sigmoid', 'tanh').
        layer: Number of SkiM blocks (default: 4).
        unit: Hidden size of LSTMs (default: 256).
        segment_size: Length of each local segment K (default: 150).
        dropout: Dropout rate (default: 0.1).
        mem_type: Memory state type for Mem-LSTM ('hc', 'h', 'c', 'id', None).
    """

    def __init__(
        self,
        input_dim: int = 256,
        causal: bool = False,
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "relu",
        layer: int = 4,
        unit: int = 256,
        segment_size: int = 150,
        dropout: float = 0.1,
        mem_type: str = "hc",
    ):
        super().__init__()
        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.segment_size = segment_size
        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk

        # SkiM Model Engine
        self.skim = SkiM(
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim * self.num_outputs,
            dropout=dropout,
            num_blocks=layer,
            segment_size=segment_size,
            bidirectional=not causal,
            mem_type=mem_type,
            norm_type="cLN" if causal else "gLN",
        )

        # Nonlinearity for estimated masks
        if nonlinear == "sigmoid":
            self.nonlinear = nn.Sigmoid()
        elif nonlinear == "relu":
            self.nonlinear = nn.ReLU()
        elif nonlinear == "tanh":
            self.nonlinear = nn.Tanh()
        else:
            raise ValueError(f"Unsupported nonlinear: {nonlinear}")

    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
        additional: Optional[Dict] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, OrderedDict]:
        """Forward pass for separation.

        Args:
            input: Latent features tensor (Batch, Time, Channel).
            ilens: Feature lengths (Batch,).

        Returns:
            masked: List of masked latent tensors [S1, S2, ...] each (Batch, Time, Channel).
            ilens: Feature lengths (Batch,).
            others: OrderedDict containing estimated masks.
        """
        B, T, N = input.shape
        processed = self.skim(input)  # (B, T, N * num_outputs)
        processed = processed.view(B, T, N, self.num_outputs)

        # Pisahkan masks untuk tiap speaker
        masks = self.nonlinear(processed).unbind(dim=3)
        if self.predict_noise:
            *masks, mask_noise = masks

        # Terapkan masking pada representasi laten input
        masked = [input * m for m in masks]

        others = OrderedDict(
            zip([f"mask_spk{i + 1}" for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise

        return masked, ilens, others

    @property
    def num_spk(self) -> int:
        return self._num_spk
