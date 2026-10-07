# TDNext comparison

Reviewed 2026-10-07. This records the original implementation and the evidence
behind the multiscale decoder change; see [README.md](README.md) for the resulting
API. The repository describes this module as a SkiM adaptation inspired by TDNext,
so it should not be presented as a complete TDNext reproduction.

## Primary evidence

Ye et al., *Addressing human speech characteristics in single-channel speaker
extraction networks*, PeerJ Computer Science 11:e3326 (2025):
[paper PDF](https://peerj.com/articles/cs-3326.pdf),
[indexed copy of the published paper](https://www.researchgate.net/publication/397209772_Addressing_human_speech_characteristics_in_single-channel_speaker_extraction_networks).
The publisher returned HTTP 403/429 during this review; the published PDF text
was accessible through the indexed copy.

- Figure 1 and Eq. (2), pp. 5–7: three ReLU convolution features `W_i`, concatenated
  for the extractor; common stride `L1/2`.
- Equations (8)–(9), p. 12: `S_i = Mask_i * W_i`, then `x_i = D_i(S_i)`, for
  three scales. Each decoder uses its matching encoder kernel length.
- Equations (10)–(11), pp. 12–13: separate waveform SI-SDR losses plus speaker
  classification cross-entropy. Tables 1 and 5 use scale weights `(0.8, 0.1, 0.1)`
  and CE weight `10`. These are loss weights, not waveform fusion weights.
- Table 4, p. 17: final TDNext uses `(20, 40, 80)` samples at 8 kHz; the initial
  configuration is `(20, 80, 160)`.
- Figures 1–4: target-speaker reference conditioning, a spectral reference branch,
  Spk/SE blocks, and TD-ConvNeXt/TCN extraction.
- The decoder section selects by “minimum discrimination rate” without defining
  an executable selection criterion. The encoding description does not fully
  specify padding for unequal kernels. [Source: published paper above.]

The paper's Data Availability section points to
[the authors' archive](https://doi.org/10.5281/zenodo.15761810).
The accompanying [TDNext repository](https://github.com/yxhjxvtc/TDNext) was
inspected at commit `869f5210ac2c0f883f43e0fe9f83003c177cb944`:

- Inside `TDNext.zip`, `nnet/TDNext.py` retains `w1/w2/w3`, predicts separate
  masks, and uses three transposed convolutions. Normalization precedes the
  extractor's 1×1 projection. Longer encoder branches receive right padding to
  match the short branch's frame count.
- `libs/trainer.py::SiSnrTrainer.compute_loss` supervises all three decoded
  waveforms with weights `0.8/0.1/0.1` and adds `10 * CE`.
- `decode_0_tt.py::NnetComputer.compute` returns the first (short) waveform.
  This is useful inference evidence, although the script retains an import of
  `nnet.spex_plus`, absent from the archive. The released `TDNext.py` also lacks
  the paper's spectral reference branch. The archive is therefore not an exact,
  verified runnable specification of every published component.
  [Source: authors' repository and archive above.]

## Findings in the original repository

| Area | Original behavior | Interpretation |
| --- | --- | --- |
| Decoder | `MultiScaleConvDecoder` contained one `ConvTranspose1d`, with kernel 16. | The name overstated its behavior; a true multiscale decoder requires separate synthesis branches. |
| Masked representation | Encoder projected `3N` channels to `N`; separator masked only that normalized fused tensor. | Separate original scale features were unavailable to matching decoders. Merely adding three kernels to the existing decoder would not implement the paper's scale-specific reconstruction path. |
| Training | The multiscale training scripts used one decoded waveform per speaker and one SI-SNR/PIT objective. | Additional synthesis branches need direct loss supervision; unused middle/long outputs cannot train their decoder weights. |
| Architecture/task | SegLSTM/MemLSTM separate two or three speakers without enrollment speech. | Intentional SkiM blind-separation adaptation. Adding a reference encoder or replacing SkiM would change the task and research design. |
| Windows | Kernels `(16, 32, 64)`, stride 8, training at 16 kHz. | These are 1/2/4 ms windows. Retaining the paper's final window durations at 16 kHz would instead require `(40, 80, 160)`, stride 20. Window selection remains an explicit experiment choice. |
| Encoder lengths | `flens` was filled with the batch maximum frame count. | Incorrect for batches with different valid audio lengths. |
| Short inputs/tails | Padding gave `floor(samples/stride)` frames; inputs shorter than one stride could fail convolution. | Requires explicit handling of partial final frames and short signals. |
| Decoder alignment | Decoder cropped from sample zero despite encoder left/symmetric padding. | Reconstructed coordinates include padding; matching synthesis must remove the corresponding left padding before fitting the requested length. |
| Cumulative normalization | Variance accumulated deviations from changing prefix means independently per channel. | This is not the variance of each observed prefix, and differs from normalization over cumulative time/channel samples. |

Code references: [encoder](multiscale_encoder.py),
[separator](skim_multiscale_separator.py), [decoder](multiscale_decoder.py),
[two-speaker trainer](../../train/2speaker/skim-multiscale/train_skim_multiscale_2spk.py),
[three-speaker trainer](../../train/3speaker/skim-multiscale/train_skim_multiscale_3spk.py),
[transfer trainer](../../train/3speaker/skim-multiscale/train_skim_multiscale_3spk_transfer.py).

## Implemented result

New multiscale training configurations enable `preserve_scales=True` and
`num_scales=3`. They retain raw ReLU scale features, normalize before the
separator bottleneck, predict a separate mask for each scale/speaker, and
decode with three matching kernels. The ESPnet training integration evaluates
the weighted `(0.8, 0.1, 0.1)` waveform objective with one shared PIT assignment.
Inference selects the short output. The weights are not used to mix waveforms.

The setup wizard additionally supports user-selected kernels and independent
strides per scale. Different strides are an explicit extension beyond TDNext's
shared stride: native branch features are linearly resampled onto the finest
frame grid for SkiM, and each masked branch is resampled back to its native
frame count before matched decoding. Independent strides require noncausal mode;
the default shared-stride path performs no resampling. Encoder/decoder settings
are stored in checkpoints. Wizard experiments use separate size-specific
directories, including the corresponding two-speaker transfer source.

New encoding rounds up partial final frames and returns per-example frame
lengths. New decoding removes matching encoder left padding and masks waveform
samples beyond each valid length. The original symmetric/noncausal and
left/causal padding policies remain explicit deviations from the authors'
right-padding implementation. CPU training uses full precision after an
integration check reproduced a CPU autocast/LSTM failure; CUDA autocast stays
enabled, with waveform loss statistics evaluated in float32.

Legacy fused/single-decoder checkpoints retain their original parameter keys
and can still load for inference/evaluation. The new architecture needs a fresh
training run; it cannot directly resume an old checkpoint. The legacy causal
fusion path's cumulative normalization discrepancy is retained for checkpoint
behavior compatibility; the new path uses channel normalization instead.

Validation covers per-scale feature/mask preservation, impulse alignment with
causal and symmetric padding, short/odd audio and variable valid lengths,
independent decoder inputs, a shared PIT assignment under conflicting scale
predictions, gradients through all analysis/synthesis branches, production
builders including two-to-three-speaker transfer, and new/legacy checkpoints
through inference and evaluation loaders. These checks ran on CPU with ESPnet
202304; CUDA execution and separation quality on trained data were not tested.

Preserving SkiM, the dataset, PIT, and the repository's sampling rate is a scoped
adaptation. A paper-style reconstruction path should preserve scale features,
apply scale-specific masks, decode each matching scale, and supervise each
waveform. Short-scale inference follows the released decoding script; it should
not be described as an unambiguous requirement of the paper's prose. Structural
and gradient checks establish implementation behavior, not reproduction of the
paper's reported separation scores.
