import sys
import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from implementation.skim_multiscale.multiscale_encoder import MultiScaleConvEncoder
from implementation.skim_multiscale.skim_multiscale import SkiM, SegLSTM, MemLSTM
from implementation.skim_multiscale.skim_multiscale_separator import (
    SkiMMultiScaleSeparator,
)
from implementation.skim_multiscale.multiscale_decoder import MultiScaleConvDecoder


class TestSkiMMultiScale(unittest.TestCase):
    """Test suite for SkiM with Multi-Scale Encoder Architecture."""

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 2
        self.samples = 16000  # 1 second of 16 kHz audio
        self.dummy_audio = torch.randn(self.batch_size, self.samples)
        self.ilens = torch.tensor([self.samples, self.samples], dtype=torch.long)

    def test_multiscale_encoder_forward_and_shapes(self):
        """Verifikasi dimensi output dan konsistensi frame MultiScaleConvEncoder."""
        encoder = MultiScaleConvEncoder(
            channel=128,
            out_channel=128,
            kernel_sizes=(16, 32, 64),
            stride=8,
            causal=False,
            nonlinear="relu",
        )
        feature, flens = encoder(self.dummy_audio, self.ilens)

        expected_frames = self.samples // 8
        self.assertEqual(feature.dim(), 3)
        self.assertEqual(feature.size(0), self.batch_size)
        self.assertAlmostEqual(feature.size(1), expected_frames, delta=4)
        self.assertEqual(feature.size(2), 128)
        self.assertEqual(flens.size(0), self.batch_size)
        self.assertFalse(torch.isnan(feature).any())

    def test_multiscale_encoder_causal_mode(self):
        """Verifikasi encoder dalam mode kausal (streaming low-latency)."""
        encoder_causal = MultiScaleConvEncoder(
            channel=64,
            out_channel=64,
            kernel_sizes=(16, 32, 64),
            stride=8,
            causal=True,
            norm_type="cLN",
        )
        feature, flens = encoder_causal(self.dummy_audio, self.ilens)
        self.assertEqual(feature.size(0), self.batch_size)
        self.assertEqual(feature.size(2), 64)
        self.assertFalse(torch.isnan(feature).any())

    def test_multiscale_encoder_gradients(self):
        """Pastikan gradien mengalir ke seluruh cabang konvolusi dan layer proyeksi."""
        encoder = MultiScaleConvEncoder(
            channel=32,
            kernel_sizes=(16, 32, 64),
            stride=8,
        )
        x = self.dummy_audio.clone().requires_grad_(True)
        feature, _ = encoder(x, self.ilens)
        loss = feature.sum()
        loss.backward()

        self.assertIsNotNone(x.grad)
        for i, conv in enumerate(encoder.conv_branches):
            self.assertIsNotNone(
                conv.weight.grad, f"Gradien cabang conv {i} tidak mengalir"
            )
            self.assertGreater(conv.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(encoder.proj.weight.grad)

    def test_skim_separator_2speakers(self):
        """Verifikasi SkiMMultiScaleSeparator untuk pemisahan 2 pembicara."""
        T_frames = 150
        input_dim = 64
        dummy_feature = torch.randn(self.batch_size, T_frames, input_dim)
        flens = torch.full((self.batch_size,), T_frames, dtype=torch.long)

        separator = SkiMMultiScaleSeparator(
            input_dim=input_dim,
            causal=False,
            num_spk=2,
            nonlinear="relu",
            layer=2,
            unit=64,
            segment_size=50,
            dropout=0.0,
            mem_type="hc",
        )

        masked_list, out_flens, others = separator(dummy_feature, flens)

        self.assertEqual(len(masked_list), 2)
        self.assertEqual(masked_list[0].shape, (self.batch_size, T_frames, input_dim))
        self.assertEqual(masked_list[1].shape, (self.batch_size, T_frames, input_dim))
        self.assertIn("mask_spk1", others)
        self.assertIn("mask_spk2", others)
        self.assertFalse(torch.isnan(masked_list[0]).any())
        self.assertFalse(torch.isnan(masked_list[1]).any())

    def test_skim_separator_3speakers(self):
        """Verifikasi SkiMMultiScaleSeparator untuk pemisahan 3 pembicara."""
        T_frames = 100
        input_dim = 64
        dummy_feature = torch.randn(self.batch_size, T_frames, input_dim)
        flens = torch.full((self.batch_size,), T_frames, dtype=torch.long)

        separator = SkiMMultiScaleSeparator(
            input_dim=input_dim,
            causal=False,
            num_spk=3,
            layer=2,
            unit=64,
            segment_size=50,
        )

        masked_list, _, others = separator(dummy_feature, flens)
        self.assertEqual(len(masked_list), 3)
        self.assertIn("mask_spk3", others)

    def test_multiscale_decoder_shape(self):
        """Verifikasi rekonstruksi waktu pada MultiScaleConvDecoder."""
        T_frames = 2000
        channel = 128
        dummy_latent = torch.randn(self.batch_size, T_frames, channel)

        decoder = MultiScaleConvDecoder(
            channel=channel,
            kernel_size=16,
            stride=8,
        )
        wav, olens = decoder(dummy_latent, self.ilens)

        self.assertEqual(wav.dim(), 2)
        self.assertEqual(wav.size(0), self.batch_size)
        self.assertEqual(wav.size(1), self.samples)
        self.assertFalse(torch.isnan(wav).any())

    def test_end_to_end_pipeline(self):
        """Verifikasi alur lengkap: Waveform -> MultiScaleEncoder -> SkiM -> Decoder -> Waveforms."""
        channel = 64
        encoder = MultiScaleConvEncoder(
            channel=channel,
            out_channel=channel,
            kernel_sizes=(16, 32, 64),
            stride=8,
        )
        separator = SkiMMultiScaleSeparator(
            input_dim=channel,
            causal=False,
            num_spk=2,
            layer=2,
            unit=64,
            segment_size=50,
        )
        decoder = MultiScaleConvDecoder(
            channel=channel,
            kernel_size=16,
            stride=8,
        )

        # 1. Encode
        feats, flens = encoder(self.dummy_audio, self.ilens)

        # 2. Separate
        masked, _, others = separator(feats, flens)

        # 3. Decode
        est_sources = []
        for m in masked:
            rec_wav, _ = decoder(m, self.ilens)
            est_sources.append(rec_wav)

        self.assertEqual(len(est_sources), 2)
        for s in est_sources:
            self.assertEqual(s.shape, self.dummy_audio.shape)
            self.assertFalse(torch.isnan(s).any())

        # 4. Backward loss test (Dummy SI-SNR loss proxy)
        dummy_loss = (est_sources[0] ** 2).mean() + (est_sources[1] ** 2).mean()
        dummy_loss.backward()
        self.assertIsNotNone(encoder.conv_branches[0].weight.grad)
        self.assertIsNotNone(decoder.conv_trans1d.weight.grad)

    def test_variable_audio_lengths(self):
        """Uji stabilitas pada berbagai panjang sampel audio (panjang ganjil/tidak rata)."""
        channel = 32
        encoder = MultiScaleConvEncoder(channel=channel, stride=8)
        decoder = MultiScaleConvDecoder(channel=channel, kernel_size=16, stride=8)

        for length in [7999, 12345, 24001]:
            audio = torch.randn(1, length)
            ilens = torch.tensor([length], dtype=torch.long)
            feat, flens = encoder(audio, ilens)
            rec, _ = decoder(feat, ilens)
            self.assertEqual(rec.shape, (1, length), f"Gagal pada panjang audio {length}")
            self.assertFalse(torch.isnan(rec).any())

    def test_paper_kernel_configurations(self):
        """Uji konfigurasi kernel multi-skala sesuai paper SpEx+/TDNext: (20, 40, 80) dengan stride 10."""
        channel = 64
        encoder = MultiScaleConvEncoder(
            channel=channel,
            out_channel=channel,
            kernel_sizes=(20, 40, 80),
            stride=10,
        )
        audio = torch.randn(2, 16000)
        ilens = torch.tensor([16000, 16000], dtype=torch.long)
        feat, flens = encoder(audio, ilens)

        self.assertEqual(feat.size(2), channel)
        self.assertAlmostEqual(feat.size(1), 1600, delta=5)
        self.assertFalse(torch.isnan(feat).any())

    def test_optimization_step(self):
        """Pastikan gradient descent update bobot model bekerja secara normal tanpa runtime error."""
        channel = 32
        encoder = MultiScaleConvEncoder(channel=channel, kernel_sizes=(16, 32, 64), stride=8)
        separator = SkiMMultiScaleSeparator(input_dim=channel, layer=1, unit=32, segment_size=20)
        decoder = MultiScaleConvDecoder(channel=channel, kernel_size=16, stride=8)

        params = list(encoder.parameters()) + list(separator.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3)

        initial_weight = encoder.conv_branches[0].weight.clone()

        # Step 1
        optimizer.zero_grad()
        feat, flens = encoder(self.dummy_audio, self.ilens)
        masked, _, _ = separator(feat, flens)
        out1, _ = decoder(masked[0], self.ilens)
        loss = (out1 - self.dummy_audio).abs().mean()
        loss.backward()
        optimizer.step()

        updated_weight = encoder.conv_branches[0].weight
        self.assertFalse(torch.equal(initial_weight, updated_weight), "Bobot harus berubah setelah optimization step")

    def test_transfer_learning_weight_compatibility(self):
        """Verifikasi transfer bobot dari model 2-speaker ke 3-speaker tanpa error shape mismatch."""
        channel = 32
        # Model sumber 2-speaker
        enc_2spk = MultiScaleConvEncoder(channel=channel, stride=8)
        sep_2spk = SkiMMultiScaleSeparator(input_dim=channel, num_spk=2, layer=2, unit=32)

        # Model tujuan 3-speaker
        enc_3spk = MultiScaleConvEncoder(channel=channel, stride=8)
        sep_3spk = SkiMMultiScaleSeparator(input_dim=channel, num_spk=3, layer=2, unit=32)

        # Transfer bobot encoder
        enc_3spk.load_state_dict(enc_2spk.state_dict())
        self.assertTrue(torch.equal(enc_3spk.conv_branches[0].weight, enc_2spk.conv_branches[0].weight))

        # Transfer bobot separator layer-by-layer (partial transfer)
        src_dict = sep_2spk.state_dict()
        tgt_dict = sep_3spk.state_dict()

        transferred_count = 0
        reinit_count = 0
        for k, v in src_dict.items():
            if k in tgt_dict:
                if tgt_dict[k].shape == v.shape:
                    tgt_dict[k] = v
                    transferred_count += 1
                else:
                    reinit_count += 1

        sep_3spk.load_state_dict(tgt_dict)

        # Pastikan core LSTM ditransfer sedangkan output_fc direinisialisasi
        self.assertGreater(transferred_count, 0)
        self.assertGreater(reinit_count, 0)  # output_fc layer shape berbeda (2spk vs 3spk)

        # Pastikan model 3-speaker dapat beroperasi normal setelah transfer
        feat, flens = enc_3spk(self.dummy_audio, self.ilens)
        masked_3spk, _, _ = sep_3spk(feat, flens)
        self.assertEqual(len(masked_3spk), 3)

    def test_scale_reconstruction_alignment_and_lengths(self):
        """Impulse coordinates survive each scale's analysis/synthesis padding."""
        for causal in (False, True):
            encoder = MultiScaleConvEncoder(channel=1, preserve_scales=True, causal=causal)
            decoder = MultiScaleConvDecoder(channel=1, kernel_sizes=(16, 32, 64), causal=causal)
            for enc, dec, k in zip(encoder.conv_branches, decoder.deconv_branches, encoder.kernel_sizes):
                left = k - 8 if causal else (k - 8) // 2
                with torch.no_grad():
                    enc.weight.zero_()
                    dec.weight.zero_()
                    enc.weight[0, 0, left] = 1
                    dec.weight[0, 0, left] = 1
            for length in (1, 7, 8, 9, 17, 65):
                with self.subTest(causal=causal, length=length):
                    audio = torch.zeros(2, length)
                    audio[:, ::8] = 1
                    lengths = torch.tensor([length, max(1, length - 8)])
                    audio[1, lengths[1]:] = 0
                    features, flens = encoder(audio, lengths)
                    self.assertEqual(features.shape, (2, (length + 7) // 8, 3))
                    torch.testing.assert_close(flens, (lengths + 7) // 8)
                    waveforms, olens = decoder.forward_scales(features, lengths)
                    torch.testing.assert_close(olens, lengths)
                    self.assertEqual(len(waveforms), 3)
                    for waveform in waveforms:
                        torch.testing.assert_close(waveform, audio)
                    selected, _ = decoder(features, lengths)
                    torch.testing.assert_close(selected, waveforms[0])

    def test_scale_masks_preserve_original_features(self):
        """Fusion belongs only to mask estimation; each mask multiplies W_i."""
        encoder = MultiScaleConvEncoder(channel=4, preserve_scales=True)
        audio = torch.randn(2, 65)
        features, flens = encoder(audio, torch.tensor([65, 49]))
        self.assertEqual(encoder.output_dim, 12)
        self.assertTrue((features >= 0).all())
        separator = SkiMMultiScaleSeparator(input_dim=4, num_scales=3, layer=1, unit=4, segment_size=4)
        masked, _, others = separator(features, flens)
        for s, source in enumerate(masked, 1):
            torch.testing.assert_close(source, features * others[f"mask_spk{s}"])
            self.assertEqual(source.shape[-1], 12)

    def test_multiscale_shared_pit(self):
        """Conflicting scale assignments must not get independent zero losses."""
        from espnet2.enh.loss.criterions.time_domain import TimeDomainMSE
        from implementation.skim_multiscale.multiscale_model import MultiScalePITSolver

        refs = [torch.zeros(2, 8), torch.ones(2, 8)]
        estimates = [
            [refs[0].clone(), refs[1].clone()],
            [refs[1].clone(), refs[0].clone()],
            [refs[1].clone(), refs[0].clone()],
        ]
        solver = MultiScalePITSolver(TimeDomainMSE())
        loss, _, others = solver(refs, estimates)
        torch.testing.assert_close(loss, torch.tensor(0.2))
        torch.testing.assert_close(others["perm"], torch.tensor([[0, 1], [0, 1]]))

        # Changing only padding must not change the objective.
        for scale in estimates:
            for estimate in scale:
                estimate[1, 4:] = 100
        lengths = torch.tensor([8, 4])
        loss, _, _ = solver(refs, estimates, {"speech_lengths": lengths})
        torch.testing.assert_close(loss, torch.tensor(0.2))

    def test_multiscale_espnet_loss_and_gradients(self):
        """The real ESPnet forward/loss must train every analysis/synthesis branch."""
        from espnet2.enh.loss.criterions.time_domain import SISNRLoss
        from implementation.skim_multiscale.multiscale_model import (
            MultiScaleEnhancementModel, MultiScalePITSolver,
        )

        for num_spk in (2, 3):
            with self.subTest(num_spk=num_spk):
                encoder = MultiScaleConvEncoder(channel=8, preserve_scales=True)
                separator = SkiMMultiScaleSeparator(
                    input_dim=8, num_scales=3, num_spk=num_spk, layer=2,
                    unit=8, segment_size=4, nonlinear="sigmoid", dropout=0,
                )
                decoder = MultiScaleConvDecoder(channel=8, kernel_sizes=(16, 32, 64))
                solver = MultiScalePITSolver(SISNRLoss(clamp_db=30))
                model = MultiScaleEnhancementModel(encoder, separator, decoder, None, [solver])
                refs = {f"speech_ref{s + 1}": torch.randn(2, 129) for s in range(num_spk)}
                mixture = sum(refs.values())
                loss, stats, weight = model(mixture, torch.tensor([129, 113]), **refs)
                self.assertTrue(torch.isfinite(loss))
                self.assertIn("loss", stats)
                self.assertEqual(weight.item(), 2)
                loss.backward()
                for branch in list(encoder.conv_branches) + list(decoder.deconv_branches):
                    self.assertIsNotNone(branch.weight.grad)
                    self.assertTrue(torch.isfinite(branch.weight.grad).all())
                    self.assertGreater(branch.weight.grad.abs().sum().item(), 0)
                self.assertGreater(separator.input_proj.weight.grad.abs().sum().item(), 0)

    def test_decoder_scale_isolation(self):
        decoder = MultiScaleConvDecoder(channel=4, kernel_sizes=(16, 32, 64))
        features = torch.randn(1, 5, 12)
        lengths = torch.tensor([33])
        original, _ = decoder.forward_scales(features, lengths)
        features[..., 4:8] = 0
        changed, _ = decoder.forward_scales(features, lengths)
        torch.testing.assert_close(original[0], changed[0])
        torch.testing.assert_close(original[2], changed[2])
        self.assertFalse(torch.equal(original[1], changed[1]))

    def test_independent_strides_alignment_and_gradients(self):
        for strides in ((8, 16, 32), (12, 8, 24), (7, 11, 13)):
            with self.subTest(strides=strides):
                encoder = MultiScaleConvEncoder(channel=4, stride=strides, preserve_scales=True)
                decoder = MultiScaleConvDecoder(channel=4, kernel_sizes=(16, 32, 64), stride=strides)
                for conv, deconv, stride in zip(encoder.conv_branches, decoder.deconv_branches, strides):
                    self.assertEqual(conv.stride, (stride,))
                    self.assertEqual(deconv.stride, (stride,))
                for length in (1, 9, 65):
                    audio = torch.randn(2, length, requires_grad=True)
                    lengths = torch.tensor([length, max(1, length // 2)])
                    features, flens = encoder(audio, lengths)
                    self.assertEqual(features.size(1), (length + min(strides) - 1) // min(strides))
                    torch.testing.assert_close(flens, (lengths + min(strides) - 1) // min(strides))
                    waveforms, _ = decoder.forward_scales(features, lengths)
                    for waveform in waveforms:
                        self.assertEqual(waveform.shape, audio.shape)
                        self.assertTrue(torch.isfinite(waveform).all())
                        self.assertTrue((waveform[1, lengths[1]:] == 0).all())
                    sum(w.square().mean() for w in waveforms).backward()
                    self.assertTrue(torch.isfinite(audio.grad).all())
                for branch in list(encoder.conv_branches) + list(decoder.deconv_branches):
                    self.assertGreater(branch.weight.grad.abs().sum().item(), 0)

    def test_multiscale_settings_validation(self):
        from utils.multiscale_config import parse_multiscale_settings

        self.assertEqual(parse_multiscale_settings('40/80/160', '10/20/40'),
                         ((40, 80, 160), (10, 20, 40)))
        self.assertEqual(parse_multiscale_settings('16/32/64', '8'),
                         ((16, 32, 64), (8, 8, 8)))
        for kernels, strides in (('1/2', '1'), ('32/16/64', '8'),
                                 ('16/32/64', '0'), ('16/32/64', '8/16'),
                                 ('16/32/64', '17/16/32')):
            with self.assertRaises(ValueError):
                parse_multiscale_settings(kernels, strides)

    def test_training_configs_and_checkpoint_loaders(self):
        """Exercise the real builders and both checkpoint consumers, old and new."""
        def load_script(relative):
            path = PROJECT_ROOT / relative
            spec = importlib.util.spec_from_file_location(path.stem, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        inference = load_script("inference/separate.py")
        evaluation = load_script("eval/run_eval.py")
        device = torch.device("cpu")
        evaluation.device = device
        scripts = (
            "train/2speaker/skim-multiscale/train_skim_multiscale_2spk.py",
            "train/3speaker/skim-multiscale/train_skim_multiscale_3spk.py",
            "train/3speaker/skim-multiscale/train_skim_multiscale_3spk_transfer.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts:
                with self.subTest(script=script):
                    with mock.patch.dict(os.environ, {
                        'TSS_MULTISCALE_KERNELS': '40/80/160',
                        'TSS_MULTISCALE_STRIDES': '10/20/40',
                    }):
                        trainer = load_script(script)
                    cfg = trainer.MODEL_CONFIG
                    self.assertTrue(cfg["encoder"]["preserve_scales"])
                    for component in ('encoder', 'decoder'):
                        self.assertEqual(cfg[component]['kernel_sizes'], (40, 80, 160))
                        self.assertEqual(cfg[component]['stride'], (10, 20, 40))
                    self.assertEqual(cfg["separator"]["num_scales"], len(cfg["decoder"]["kernel_sizes"]))
                    # Keep the production builder/config path, with a small CPU model.
                    cfg["encoder"]["channel"] = 8
                    cfg["decoder"]["channel"] = 8
                    cfg["separator"].update(input_dim=8, unit=8, layer=2, segment_size=4)
                    with contextlib.redirect_stdout(io.StringIO()):
                        if "transfer" in script:
                            model = trainer.build_model(device, Path(directory) / "2spk.pth")
                        else:
                            model = trainer.build_model(device)
                    mixture = torch.randn(1, 129)
                    refs = {f"speech_ref{i + 1}": torch.randn_like(mixture) for i in range(model.num_spk)}
                    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                        loss, _, _ = model(mixture, torch.tensor([129]), **refs)
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    for branch in model.decoder.deconv_branches:
                        self.assertTrue(torch.isfinite(branch.weight.grad).all())

                    checkpoint = Path(directory) / "new.pth"
                    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, checkpoint)
                    if model.num_spk == 2:
                        torch.save({"model_state_dict": model.state_dict(), "config": cfg}, Path(directory) / "2spk.pth")
                    enc, sep, dec, num_spk, arch = inference.load_model(checkpoint, device)
                    self.assertEqual((num_spk, arch), (model.num_spk, "multiscale"))
                    actual = inference.separate(enc, sep, dec, mixture.numpy()[0], device)
                    # Dropout is disabled in loaded models.
                    model.eval()
                    expected = inference.separate(model.encoder, model.separator, model.decoder, mixture.numpy()[0], device)
                    for want, got in zip(expected, actual):
                        torch.testing.assert_close(torch.from_numpy(want), torch.from_numpy(got))
                    enc, sep, dec = evaluation.build_model(num_spk, arch, checkpoint)
                    actual = evaluation.separate(enc, sep, dec, mixture.numpy()[0])
                    for want, got in zip(expected, actual):
                        torch.testing.assert_close(torch.from_numpy(want), torch.from_numpy(got))

            legacy_cfg = evaluation.build_config(2, "multiscale")
            legacy_cfg["encoder"].update(channel=8, out_channel=8)
            legacy_cfg["decoder"]["channel"] = 8
            legacy_cfg["separator"].update(input_dim=8, unit=8, layer=1, segment_size=4)
            enc = MultiScaleConvEncoder(**legacy_cfg["encoder"])
            sep_cfg = {k: v for k, v in legacy_cfg["separator"].items() if k in inference._MULTISCALE_KEYS}
            sep = SkiMMultiScaleSeparator(**sep_cfg)
            dec = MultiScaleConvDecoder(**legacy_cfg["decoder"])
            modules = {"encoder": enc, "separator": sep, "decoder": dec}
            state = {f"{name}.{k}": v for name, module in modules.items() for k, v in module.state_dict().items()}
            checkpoint = Path(directory) / "old.pth"
            torch.save({"model_state_dict": state, "config": legacy_cfg}, checkpoint)
            loaded = inference.load_model(checkpoint, device)[:3]
            evaluated = evaluation.build_model(2, "multiscale", checkpoint)
            for triplet in (loaded, evaluated):
                for original, restored in zip(modules.values(), triplet):
                    self.assertEqual(original.state_dict().keys(), restored.state_dict().keys())
                    for k, value in original.state_dict().items():
                        torch.testing.assert_close(value, restored.state_dict()[k])


if __name__ == "__main__":
    unittest.main()
