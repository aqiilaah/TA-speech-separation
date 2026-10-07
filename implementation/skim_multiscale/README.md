# SkiM with Multi-Scale Encoder and Decoder

Implementasi modifikasi arsitektur **SkiM (Skipping Memory LSTM)** dengan front-end **Multi-Scale 1D Convolutional Encoder**, terinspirasi dari prinsip multi-resolusi akustik pada paper **TDNext** (Ye et al., PeerJ Comp. Sci. 2025) dan **SpEx+** (Ge et al., Interspeech 2020), dipadukan dengan pemodelan sekuens efisien dari paper **SkiM** (Li et al., ICASSP 2022).

Ini adalah adaptasi SkiM untuk pemisahan dua/tiga pembicara tanpa reference
speech, bukan reproduksi penuh TDNext. Perbandingan dengan paper, persamaan,
dan kode penulis tersedia dalam [PAPER_REVIEW.md](PAPER_REVIEW.md).

---

## 1. Motivasi & Konsep Arsitektur

- **Encoder Konvensional:** Menggunakan ukuran *kernel* tunggal (misal 16 sampel = 1 ms). Terjadi kompromi antara resolusi waktu dan frekuensi:
  - Kernel pendek: Resolusi temporal tinggi (bagus untuk transien, onset, konsonan), tetapi resolusi frekuensi rendah.
  - Kernel panjang: Resolusi frekuensi tinggi (bagus untuk harmonik, pitch, formant vokal), tetapi resolusi temporal rendah.
- **Multi-Scale Encoder:** Menggabungkan 3 cabang konvolusi 1D paralel dengan ukuran kernel berbeda:
  - **Short ($L_1 = 16$):** Menangkap transien cepat.
  - **Middle ($L_2 = 32$):** Menangkap dinamika fonem menengah.
  - **Long ($L_3 = 64$):** Menangkap harmonik & pitch period.
- **Stride:** Default semua cabang memakai stride 8, sehingga frame ketiga
  skala identik. Stride independen seperti `(8,16,32)` juga didukung pada mode
  nonkausal: encoder menginterpolasi fitur ke grid frame stride terkecil,
  kemudian decoder mengembalikan fitur tiap skala ke jumlah frame stride
  aslinya sebelum rekonstruksi. Pilihan ini merupakan perluasan adaptasi;
  TDNext menggunakan stride bersama. `causal=True` memakai stride bersama.
- **Feature Fusion:** Encoder mempertahankan fitur asli tiga skala sebagai
  $(B,T,3N)$. Di separator, normalisasi kanal diikuti proyeksi per frame
  $3N\to N$ hanya untuk jalur estimasi mask.
- **SkiM Separator:** SkiM memproses bottleneck $(B,T,N)$ dengan `SegLSTM`
  dan `MemLSTM`, lalu menghasilkan mask per skala untuk setiap pembicara.
  Mask diterapkan pada fitur asli skala terkait: $S_i=M_i\odot W_i$.
- **Multi-Scale Decoder:** Tiga `ConvTranspose1d` memakai kernel yang sama
  dengan encoder. Padding kiri encoder dihapus saat decoding. Audio ganjil
  dan audio lebih pendek dari stride tetap menghasilkan panjang yang benar.
- **Training:** Ketiga waveform dilatih memakai loss SI-SNR dengan bobot
  `(0.8, 0.1, 0.1)`, mengikuti Eq. (11) TDNext. PIT memilih satu permutasi
  pembicara yang sama untuk ketiga skala. Bobot ini adalah bobot loss.
- **Inference:** `forward()` decoder mengembalikan waveform skala pendek,
  mengikuti skrip decoding penulis. `forward_scales()` mengembalikan seluruh
  skala untuk supervisi training. Log validasi training menunjukkan objective
  multiskala; evaluasi WAV memakai hasil skala pendek.

Konfigurasi `(16,32,64)` pada 16 kHz mempertahankan pilihan eksperimen repo
(1/2/4 ms). Paper TDNext terakhir memakai `(20,40,80)` pada 8 kHz
(2.5/5/10 ms); untuk mempertahankan durasi tersebut pada 16 kHz gunakan
`kernel_sizes=(40,80,160)` dan `stride=20` pada encoder dan decoder.
Padding simetris mode nonkausal tetap merupakan pilihan adaptasi repo;
kode penulis memakai padding kanan pada cabang yang lebih panjang.

Wizard `bash setup_and_train.sh --configure` menanyakan kernel dan stride
khusus pilihan model 7–9. Input memakai format `40/80/160` dan `10/20/40`;
stride tunggal seperti `20` juga diterima. Pilihan berlaku untuk encoder,
decoder, dan sumber pretraining transfer, lalu disimpan dalam
`setup_and_train.env`. Folder checkpoint wizard menyertakan kedua pilihan,
misalnya `skim-multiscale-k40_80_160-s10_20_40`.

Training manual dapat memakai variabel lingkungan yang sama:

```bash
TSS_MULTISCALE_KERNELS=40/80/160 TSS_MULTISCALE_STRIDES=10/20/40 \
python train/2speaker/skim-multiscale/train_skim_multiscale_2spk.py \
    --checkpoint-dir checkpoints/2speaker/skim-multiscale-k40_80_160-s10_20_40
```

---

## 2. Struktur Modul

Folder `implementation/skim_multiscale/`:
```text
implementation/skim_multiscale/
├── __init__.py                     # Interface modul utama
├── multiscale_encoder.py           # MultiScaleConvEncoder (Short, Mid, Long)
├── skim_multiscale.py              # Inti SkiM (SegLSTM, MemLSTM, SkiM engine)
├── skim_multiscale_separator.py    # SkiMMultiScaleSeparator (Estimasi mask multi-speaker)
├── multiscale_decoder.py           # MultiScaleConvDecoder (Rekonstruksi audio domain waktu)
├── multiscale_model.py             # Integrasi ESPnet dan loss multiskala/PIT
├── PAPER_REVIEW.md                 # Hasil verifikasi paper dan batas adaptasi
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
    preserve_scales=True,
)
separator = SkiMMultiScaleSeparator(
    input_dim=256,
    num_scales=3,
    num_spk=2,
    layer=4,
    unit=256,
    segment_size=150,
)
decoder = MultiScaleConvDecoder(channel=256, kernel_sizes=(16, 32, 64), stride=8)

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

Ketiga skrip training `skim-multiscale` sudah menggunakan
`MultiScaleEnhancementModel` dan `MultiScalePITSolver`. Menggunakan model
ESPnet standar hanya dengan `decoder.forward()` akan melatih skala pendek
saja; gunakan integrasi ini untuk loss ketiga skala.

Checkpoint lama tetap dapat dibuka untuk inference/evaluasi melalui config
yang tersimpan: encoder tanpa `preserve_scales` memakai jalur fusion lama,
separator tanpa `num_scales` memakai satu representasi, dan decoder dengan
`kernel_size` memakai satu kernel. API decoder tanpa `kernel_sizes` juga
tetap memakai jalur lama. Arsitektur baru membutuhkan training baru;
checkpoint lama tidak dapat di-resume langsung ke arsitektur baru. Wizard
memakai folder konfigurasi sendiri dan mempertahankan checkpoint lama.
Saat menjalankan training manual dengan ukuran berbeda, gunakan juga folder
checkpoint berbeda. Config ukuran encoder/decoder disimpan dalam checkpoint,
sehingga inference dan evaluasi tidak perlu variabel lingkungan training.

---

## 4. Pengujian (Unit Tests)

Jalankan test suite untuk memvalidasi dimensi tensor, stabilitas numerik, gradien backward pass, dan variasi panjang audio:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_skim_multiscale.py' -v
```
