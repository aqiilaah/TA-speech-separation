# SkiM with Multi-Scale Encoder

Implementasi modifikasi arsitektur **SkiM (Skipping Memory LSTM)** dengan front-end **Multi-Scale 1D Convolutional Encoder**, terinspirasi dari prinsip multi-resolusi akustik pada paper **TDNext** (Ye et al., PeerJ Comp. Sci. 2025) dan **SpEx+** (Ge et al., Interspeech 2020), dipadukan dengan pemodelan sekuens efisien dari paper **SkiM** (Li et al., ICASSP 2022).

---

## 1. Motivasi & Konsep Arsitektur

- **Encoder Konvensional:** Menggunakan ukuran *kernel* tunggal (misal 16 sampel = 1 ms). Terjadi kompromi antara resolusi waktu dan frekuensi:
  - Kernel pendek: Resolusi temporal tinggi (bagus untuk transien, onset, konsonan), tetapi resolusi frekuensi rendah.
  - Kernel panjang: Resolusi frekuensi tinggi (bagus untuk harmonik, pitch, formant vokal), tetapi resolusi temporal rendah.
- **Multi-Scale Encoder:** Menggabungkan 3 cabang konvolusi 1D paralel dengan ukuran kernel berbeda:
  - **Short ($L_1 = 16$):** Menangkap transien cepat.
  - **Middle ($L_2 = 32$):** Menangkap dinamika fonem menengah.
  - **Long ($L_3 = 64$):** Menangkap harmonik & pitch period.
- **Uniform Stride:** Seluruh cabang konvolusi menggunakan *stride* yang sama (misal `stride = 8`), sehingga panjang sekuens waktu ($T$) dari ketiga skala tepat identik ($W_1, W_2, W_3 \in \mathbb{R}^{B \times T \times N}$).
- **Feature Fusion:** Fitur dari ketiga skala dikonkatenasikan sepanjang dimensi kanal ($3N$), kemudian diproyeksikan kembali ke dimensi laten $N$ melalui konvolusi $1 \times 1$ dan normalisasi layer.
- **SkiM Separator:** Menerima representasi multi-skala gabungan $(B, T, N)$, membaginya menjadi segmen lokal berukuran $K=150$, memproses konteks lokal dengan `SegLSTM`, dan menyinkronkan memori antar-segmen dengan `MemLSTM` (skipping memory).

---

## 2. Struktur Modul

Folder `implementation/skim_multiscale/`:
```text
implementation/skim_multiscale/
├── __init__.py                     # Interface modul utama
├── multiscale_encoder.py           # MultiScaleConvEncoder (Short, Mid, Long + 1x1 Proj + Norm)
├── skim_multiscale.py              # Inti SkiM (SegLSTM, MemLSTM, SkiM engine)
├── skim_multiscale_separator.py    # SkiMMultiScaleSeparator (Estimasi mask multi-speaker)
├── multiscale_decoder.py           # MultiScaleConvDecoder (Rekonstruksi audio domain waktu)
└── README.md                       # Dokumentasi modul ini
```

---

## 3. Cara Penggunaan

### Contoh Python API:
```python
import torch
from implementation.skim_multiscale import (
    MultiScaleConvEncoder,
    SkiMMultiScaleSeparator,
    MultiScaleConvDecoder,
)

# 1. Inisialisasi komponen
encoder = MultiScaleConvEncoder(
    channel=256,
    kernel_sizes=(16, 32, 64),
    stride=8,
    causal=False,
)
separator = SkiMMultiScaleSeparator(
    input_dim=256,
    num_spk=2,
    layer=4,
    unit=256,
    segment_size=150,
)
decoder = MultiScaleConvDecoder(channel=256, kernel_size=16, stride=8)

# 2. Forward pass audio (Batch, Samples)
audio_mix = torch.randn(2, 32000)  # 2 detik audio 16 kHz
ilens = torch.tensor([32000, 32000], dtype=torch.long)

# Encode
feats, flens = encoder(audio_mix, ilens)

# Separate
masked_list, _, masks_dict = separator(feats, flens)

# Decode ke sinyal audio tiap pembicara
est_spk1, _ = decoder(masked_list[0], ilens)
est_spk2, _ = decoder(masked_list[1], ilens)

print(f"Output spk1: {est_spk1.shape}, Output spk2: {est_spk2.shape}")
```

---

## 4. Pengujian (Unit Tests)

Jalankan test suite untuk memvalidasi dimensi tensor, stabilitas numerik, gradien backward pass, dan variasi panjang audio:

```bash
python -m unittest tests/test_skim_multiscale.py -v
```
