import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN) across time and feature dimensions."""

    def __init__(self, channel_size: int, eps: float = 1e-5):
        super().__init__()
        self.channel_size = channel_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, channel_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, channel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Channel)
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN) across feature dimension only (causal)."""

    def __init__(self, channel_size: int, eps: float = 1e-5):
        super().__init__()
        self.channel_size = channel_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, channel_size))
        self.beta = nn.Parameter(torch.zeros(1, 1, channel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Channel)
        mean = x.mean(dim=2, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=2, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.gamma + self.beta


def choose_norm_layer(norm_type: str, channel_size: int) -> nn.Module:
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else:
        return nn.LayerNorm(channel_size)


class MemLSTM(nn.Module):
    """Mem-LSTM for cross-segment global state synchronization.

    Models inter-segment summary representations by taking cell states (c)
    and hidden states (h) produced at segment boundaries and updating them
    across the segment dimension S.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        mem_type: str = "hc",
        norm_type: str = "cLN",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.input_size = self.num_direction * hidden_size
        self.mem_type = mem_type

        assert mem_type in ["hc", "h", "c", "id", None], (
            f"Unsupported mem_type: {mem_type}"
        )

        if mem_type in ["hc", "h"]:
            self.h_net = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.h_norm = choose_norm_layer(norm_type, self.input_size)
            self.h_proj = (
                nn.Linear(hidden_size * self.num_direction, self.input_size)
                if hidden_size * self.num_direction != self.input_size
                else nn.Identity()
            )

        if mem_type in ["hc", "c"]:
            self.c_net = nn.LSTM(
                input_size=self.input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            )
            self.c_norm = choose_norm_layer(norm_type, self.input_size)
            self.c_proj = (
                nn.Linear(hidden_size * self.num_direction, self.input_size)
                if hidden_size * self.num_direction != self.input_size
                else nn.Identity()
            )

    def forward(
        self, hc: Tuple[torch.Tensor, torch.Tensor], S: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mem_type == "id" or self.mem_type is None:
            return hc

        h, c = hc
        # h, c shape: (num_dir, B*S, H)
        d, BS, H = h.shape
        B = BS // S

        h_seq = h.transpose(1, 0).contiguous().reshape(B, S, d * H)
        c_seq = c.transpose(1, 0).contiguous().reshape(B, S, d * H)

        if self.mem_type == "hc":
            out_h, _ = self.h_net(h_seq)
            h_seq = h_seq + self.h_norm(self.h_proj(out_h))
            out_c, _ = self.c_net(c_seq)
            c_seq = c_seq + self.c_norm(self.c_proj(out_c))
        elif self.mem_type == "h":
            out_h, _ = self.h_net(h_seq)
            h_seq = h_seq + self.h_norm(self.h_proj(out_h))
            c_seq = torch.zeros_like(c_seq)
        elif self.mem_type == "c":
            h_seq = torch.zeros_like(h_seq)
            out_c, _ = self.c_net(c_seq)
            c_seq = c_seq + self.c_norm(self.c_proj(out_c))

        # Jika kausal uni-directional, geser state satu segmen ke kanan (shift causal)
        if not self.bidirectional:
            h_shift = torch.zeros_like(h_seq)
            h_shift[:, 1:, :] = h_seq[:, :-1, :]
            c_shift = torch.zeros_like(c_seq)
            c_shift[:, 1:, :] = c_seq[:, :-1, :]
            h_seq, c_seq = h_shift, c_shift

        h_out = h_seq.reshape(B * S, d, H).transpose(1, 0).contiguous()
        c_out = c_seq.reshape(B * S, d, H).transpose(1, 0).contiguous()
        return (h_out, c_out)


class SegLSTM(nn.Module):
    """Seg-LSTM for intra-segment local sequence modeling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        norm_type: str = "cLN",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size * self.num_direction, input_size)
        self.norm = choose_norm_layer(norm_type, input_size)

    def forward(
        self, input: torch.Tensor, hc: Optional[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # input: (B*S, K, D)
        BS, K, D = input.shape
        if hc is None:
            d = self.num_direction
            h = torch.zeros(d, BS, self.hidden_size, dtype=input.dtype, device=input.device)
            c = torch.zeros(d, BS, self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            h, c = hc

        output, (h, c) = self.lstm(input, (h, c))
        output = self.dropout(output)
        output = self.proj(output)
        output = input + self.norm(output)
        return output, (h, c)


class SkiM(nn.Module):
    """Skipping Memory (SkiM) Core Architecture.

    Stacks L Seg-LSTM blocks interleaved with L-1 Mem-LSTM blocks to achieve
    efficient long-sequence modeling with 75% computational savings over DPRNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
        num_blocks: int = 4,
        segment_size: int = 150,
        bidirectional: bool = True,
        mem_type: str = "hc",
        norm_type: str = "gLN",
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.segment_size = segment_size
        self.dropout = dropout
        self.num_blocks = num_blocks
        self.mem_type = mem_type
        self.norm_type = norm_type

        # Seg-LSTMs
        self.seg_lstms = nn.ModuleList([
            SegLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                bidirectional=bidirectional,
                norm_type=norm_type,
            )
            for _ in range(num_blocks)
        ])

        # Mem-LSTMs
        if self.mem_type is not None:
            self.mem_lstms = nn.ModuleList([
                MemLSTM(
                    hidden_size=hidden_size,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    mem_type=mem_type,
                    norm_type=norm_type,
                )
                for _ in range(num_blocks - 1)
            ])
        else:
            self.mem_lstms = None

        # Output projection head: 1D Conv (1x1)
        self.output_fc = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(input_size, output_size, 1),
        )

    def _pad_feature(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        B, T, D = x.shape
        rest = (self.segment_size - (T % self.segment_size)) % self.segment_size
        if rest > 0:
            x = F.pad(x, (0, 0, 0, rest))
        return x, rest

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: (B, T, D)
        B, T, D = input.shape
        x_padded, rest = self._pad_feature(input)
        _, T_pad, _ = x_padded.shape
        S = T_pad // self.segment_size
        K = self.segment_size

        # Reshape menjadi (B*S, K, D)
        output = x_padded.contiguous().reshape(B * S, K, D)

        hc = None
        for i in range(self.num_blocks):
            output, hc = self.seg_lstms[i](output, hc)
            if self.mem_lstms is not None and i < self.num_blocks - 1:
                hc = self.mem_lstms[i](hc, S)

        # Kembalikan ke (B, T, D)
        output = output.reshape(B, S * K, D)[:, :T, :].contiguous()

        # Proyeksikan ke output_size
        output = self.output_fc(output.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
        return output
