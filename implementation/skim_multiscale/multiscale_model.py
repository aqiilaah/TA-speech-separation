"""ESPnet training integration for independently supervised synthesis scales."""

from itertools import permutations
import math

import torch

from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper


class MultiScaleEnhancementModel(ESPnetEnhancementModel):
    """Expose every decoded scale to the loss, preserving ESPnet checkpoint keys."""

    def forward_enhance(self, speech_mix, speech_lengths, additional=None):
        feature_mix, flens = self.encoder(speech_mix, speech_lengths)
        feature_pre, _, others = self.separator(feature_mix, flens, additional)
        per_speaker = [self.decoder.forward_scales(p, speech_lengths)[0] for p in feature_pre]
        # ESPnet supports a list of output stages, each containing all speakers.
        speech_pre = [list(scale) for scale in zip(*per_speaker)]
        others["speech_lengths"] = speech_lengths
        return speech_pre, feature_mix, feature_pre, others


class MultiScalePITSolver(AbsLossWrapper):
    """Weighted scale losses with a shared speaker permutation across scales.

    TDNext Eq. (11) supplies the scale weights. PIT is this repository's
    extension for blind separation: one speaker assignment minimizes the
    weighted objective, so scale branches keep consistent speaker identities.
    """

    def __init__(self, criterion, scale_weights=(0.8, 0.1, 0.1)):
        super().__init__()
        self.criterion = criterion
        self.scale_weights = tuple(scale_weights)
        if (not self.scale_weights or any(w < 0 for w in self.scale_weights)
                or not math.isclose(sum(self.scale_weights), 1.0)):
            raise ValueError("scale_weights must be non-negative and sum to one")

    def forward(self, ref, inf, others=None):
        others = others or {}
        if len(inf) != len(self.scale_weights) or any(len(scale) != len(ref) for scale in inf):
            raise ValueError("Expected one list of speaker waveforms per weighted scale")
        lengths = others.get("speech_lengths")
        sample_lengths = lengths.detach().cpu().tolist() if lengths is not None else None

        def score(reference, estimate):
            # Keep waveform means/energies in float32 under CUDA autocast.
            reference, estimate = reference.float(), estimate.float()
            if sample_lengths is None or all(length == reference.size(1) for length in sample_lengths):
                return self.criterion(reference, estimate)
            # Avoid including batch padding in waveform means and energies.
            return torch.cat([
                self.criterion(reference[b:b + 1, :length], estimate[b:b + 1, :length])
                for b, length in enumerate(sample_lengths)
            ])

        pair_losses = [[
            sum(w * score(r, scale[s]) for w, scale in zip(self.scale_weights, inf))
            for s in range(len(ref))
        ] for r in ref]
        assignments = list(permutations(range(len(ref))))
        losses = torch.stack([
            sum(pair_losses[r][s] for r, s in enumerate(assignment)) / len(ref)
            for assignment in assignments
        ], dim=1)
        best, indices = losses.min(dim=1)
        loss = best.mean()
        perm = torch.tensor(assignments, device=loss.device)[indices]
        return loss, {self.criterion.name: loss.detach()}, {"perm": perm}
