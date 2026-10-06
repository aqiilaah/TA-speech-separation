import sys
import unittest
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


if __name__ == "__main__":
    unittest.main()

