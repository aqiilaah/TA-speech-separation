# TA Speech Separation

Repositori ini merupakan implementasi tugas akhir mengenai pemisahan suara
dua dan tiga pembicara berbahasa Indonesia menggunakan **SkiM** dan
**SkiM dengan Multi-Head Self-Attention**. Model dibangun menggunakan PyTorch
dan ESPnet, kemudian dilatih pada campuran sintetis yang berasal dari korpus
TTML-IDN.

## Fitur Utama

- Pemisahan suara dua dan tiga pembicara.
- Model dasar SkiM.
- Model SkiM dengan tambahan *intra-segment Multi-Head Self-Attention*.
- Model SkiM dengan encoder dan decoder konvolusi multiskala serta loss per skala.
- *Transfer learning* dari model dua pembicara ke model tiga pembicara.
- Pembuatan campuran data latih secara *on-the-fly*.
- Evaluasi SI-SNR pada seluruh *test set*.
- Inferensi untuk satu file WAV atau satu folder.

## Struktur Repositori

```text
TA-speech-separation/
├── dataset/
│   ├── generator/          # Generator dataset statis
│   ├── raw/                # Dataset mentah TTML-IDN (tidak disertakan)
│   └── synthetic/          # Dataset sintetis (tidak disertakan)
├── implementation/
│   ├── skim/               # Implementasi SkiM
│   ├── skim_attention/     # Implementasi SkiM + Attention
│   └── skim_multiscale/    # Implementasi SkiM + Multi-Scale Encoder/Decoder
├── train/
│   ├── 2speaker/           # Pelatihan model dua pembicara (skim, skim-attention, skim-multiscale)
│   ├── 3speaker/           # Pelatihan model tiga pembicara (skim, skim-attention, skim-multiscale)
│   └── datasets_utils.py   # DynamicMixDataset dan dataset statis
├── checkpoints/            # Hasil pelatihan (dibuat otomatis)
├── eval/run_eval.py        # Evaluasi test set
└── inference/separate.py   # Inferensi audio
```

## Persiapan Lingkungan

### Setup dan pelatihan otomatis

Jalankan wizard dari checkout repositori yang sudah diperbarui:

```bash
bash setup_and_train.sh
```

Pada penggunaan pertama, wizard menanyakan folder proyek, CUDA atau CPU,
GPU, pilihan model, jumlah epoch, total jam dataset, serta URL
atau path ZIP TITML. Source ZIP default berasal dari konfigurasi proyek.
Password ZIP terenkripsi diminta secara tersembunyi jika belum tersedia;
password tidak disimpan dalam berkas konfigurasi.

Untuk model Multi-Scale (pilihan 7–9), wizard juga menanyakan kernel dan
stride dalam **sampel audio**, berurutan short/middle/long:

```text
Kernel sizes (short/middle/long) [16/32/64]: 40/80/160
Strides (short/middle/long; one value uses a shared stride) [8/8/8]: 10/20/40
```

Stride boleh berbeda untuk tiap skala; satu nilai seperti `20` berarti
`20/20/20`. Kernel harus meningkat dan setiap stride harus positif serta
tidak melebihi kernel pasangannya. Pilihan tersimpan dan dipakai juga untuk
pretraining dua pembicara pada transfer learning. Setiap konfigurasi wizard
memiliki folder checkpoint sendiri, misalnya
`checkpoints/2speaker/skim-multiscale-k40_80_160-s10_20_40/`, sehingga mengganti
kernel/stride tidak melanjutkan atau menimpa eksperimen konfigurasi lain.

Python dipilih otomatis, tanpa pertanyaan versi atau path. Jika Python
3.10/3.11 dengan dukungan `venv` tersedia, instalasi Python dilewati. Jika
belum tersedia (misalnya instance Vast.ai memakai Python 3.12), wizard
menggunakan `uv` yang sudah ada atau memasang `uv` di `~/.local/bin`, lalu
memasang Python 3.11. Python sistem dan shell profile tidak diubah.
Bootstrap ini membutuhkan `curl` dan akses internet ke Astral/GitHub.

Venv yang kompatibel digunakan kembali. Venv rusak atau memakai versi
Python lain dipindahkan ke `.venv.backup.*/original`, kemudian `.venv`
baru dibuat. Dataset dan checkpoint tidak dipindahkan atau dihapus.
Konfigurasi lama dengan path Python yang tidak sesuai diperbaiki otomatis.

Wizard menjalankan tujuh tahap: clone atau gunakan checkout yang sudah ada,
buat venv, unduh ZIP, ekstrak dataset mentah, buat **dev dan test saja**,
latih model, lalu tampilkan lokasi hasil. Split validasi disebut `dev` dalam
kode. Total jam dataset dibagi 80% train, 10% dev, dan 10% test. Default
wizard adalah **10 jam**: 8 jam train dinamis per epoch (5.760 campuran),
1 jam dev (720 campuran), dan 1 jam test (720 campuran), masing-masing
berdurasi 5 detik. Pilihan 50 jam tetap tersedia: 28.800 campuran train
per epoch, 3.600 dev, dan 3.600 test.

Ringkasan generator memisahkan durasi WAV statis dan train dinamis. Untuk
pilihan 50 jam, `Static WAV duration` adalah 10 jam (5 jam dev + 5 jam test),
`Dynamic train per epoch` adalah 40 jam, dan `Total dataset target` tetap 50 jam.

Jam tersebut dihitung dari durasi **mixture**, bukan penjumlahan durasi
seluruh source pembicara atau durasi rekaman mentah. Campuran train berubah
setiap epoch; 10 jam bukan batas kumulatif untuk seluruh proses pelatihan.
Korpus mentah tetap diunduh penuh.

Pilihan tersimpan dalam `setup_and_train.env` dan digunakan pada run berikutnya.
Untuk mengubah pilihan:

```bash
bash setup_and_train.sh --configure
```

Pilihan model mencakup sembilan konfigurasi pada bagian Pelatihan (SkiM,
SkiM + Attention, dan SkiM + Multi-Scale). Untuk transfer learning, wizard
melatih model dua pembicara terlebih dahulu jika checkpoint sumber belum ada. Run berikutnya melanjutkan checkpoint epoch lengkap yang
tersedia; run yang sudah mencapai jumlah epoch tujuan dilewati. Log disimpan
dalam `run.log` di folder checkpoint masing-masing model.

Gunakan folder proyek baru jika mengubah total jam dan folder sebelumnya
sudah berisi dev/test atau checkpoint. Wizard tidak menimpa dataset yang
berasal dari ukuran atau konfigurasi lain.
Jika pembuatan dataset oleh wizard terputus, run berikutnya membuat ulang
split tersebut dengan seed yang sama. Sebelum training, jumlah mixture dan
kesesuaian ID seluruh sumber diperiksa.

Wizard dapat disalin ke mesin Linux lain dan menjalankan clone ke folder baru.
Source clone adalah `https://github.com/aqiilaah/TA-speech-separation.git`.
Checkout hasil clone harus sudah memuat pembaruan dynamic mixing,
`--only-splits` pada kedua generator, serta CLI training terbaru. Pembaruan
lokal harus dipublikasikan ke GitHub sebelum setup melalui clone baru dapat
digunakan; wizard menghentikan proses jika checkout masih memakai train statis.
Skrip model yang dipilih, termasuk sumber pretraining untuk transfer learning,
diperiksa sebelum instalasi dependensi dan pembuatan dataset. Jika folder lama
tidak memiliki skrip Multi-Scale, perbarui checkout atau jalankan `--configure`
dan pilih checkout terbaru. Dataset dan checkpoint lama dapat tetap disimpan.

### Setup manual

Disarankan menggunakan lingkungan virtual Python.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Instal dependensi utama:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124
pip install espnet==202304 espnet_model_zoo
pip install "numpy<1.24" soundfile librosa==0.9.2 matplotlib tqdm
pip install mir_eval pesq pystoi 'gdown>=6,<7' pyzipper
```

GPU CUDA sangat disarankan untuk pelatihan. Inferensi dapat dijalankan pada
CPU, tetapi lebih lambat.

## Persiapan Dataset

### 1. Dataset mentah

Letakkan TTML-IDN dalam struktur berikut:

```text
dataset/raw/TTML-IDN/
└── Speech/
    ├── f01/
    │   ├── audio_001.wav
    │   └── ...
    ├── m01/
    │   ├── audio_001.wav
    │   └── ...
    └── ...
```

Nama folder pembicara harus tetap mengikuti dataset asli. Program mengenali
awalan `f` sebagai perempuan dan `m` sebagai laki-laki untuk pemilihan
kombinasi pembicara yang seimbang.

Lokasi dataset mentah juga dapat diatur melalui variabel lingkungan:

```bash
export TSS_RAW_DIR=/path/to/TTML-IDN
```

Nilai `TSS_RAW_DIR` harus menunjuk ke folder yang langsung berisi `Speech/`.

### 2. Dataset dev dan test statis

Data latih dibuat langsung dari rekaman mentah ketika pelatihan berjalan.
Namun, data dev dan test tetap menggunakan campuran WAV statis agar evaluasi
antar-epoch dan antar-model konsisten.

Struktur yang dibutuhkan:

```text
dataset/synthetic/
├── TITML-2spk-v2/
│   ├── dev/{mix,s1,s2}/
│   └── test/{mix,s1,s2}/
└── TITML-3spk-v2/
    ├── dev/{mix,s1,s2,s3}/
    └── test/{mix,s1,s2,s3}/
```

Gunakan generator berikut dari root repositori. Gunakan seed dan rasio yang
sama dengan pelatihan agar pembagian ujaran tetap konsisten.

```bash
python dataset/generator/titml_mix_generator_2spk.py \
    --titml-dir dataset/raw/TTML-IDN \
    --output-dir dataset/synthetic/TITML-2spk-v2 \
    --target-hours 50 \
    --train-ratio 0.8 \
    --dev-ratio 0.1 \
    --seed 42 \
    --only-splits dev test

python dataset/generator/titml_mix_generator_3spk.py \
    --titml-dir dataset/raw/TTML-IDN \
    --output-dir dataset/synthetic/TITML-3spk-v2 \
    --target-hours 50 \
    --train-ratio 0.8 \
    --dev-ratio 0.1 \
    --seed 42 \
    --only-splits dev test
```

Kedua perintah di atas hanya menghasilkan dev dan test. Tanpa `--only-splits`,
generator juga dapat menghasilkan train statis, tetapi berkas tersebut tidak
dibaca oleh skrip pelatihan.

## Cara Kerja Data Latih Dinamis

Semua skrip pelatihan menggunakan `DynamicMixDataset` untuk split train.
Pada setiap pemanggilan sampel, program akan:

1. memilih dua atau tiga pembicara dari *pool* ujaran train;
2. memilih ujaran secara acak;
3. memilih SNR secara acak dari -5 sampai +5 dB;
4. memberi offset waktu acak hingga satu detik;
5. mencampur dan menormalisasi sinyal;
6. menerapkan augmentasi kecepatan acak.

Jumlah campuran per epoch mengikuti pilihan jam pada wizard, melalui
`TSS_TRAIN_EPOCH_SIZE`. Untuk 10 jam total, satu epoch train terdiri dari
5.760 campuran. Training manual tanpa variabel tersebut mempertahankan
default lama 28.800 campuran (40 jam train dari total 50 jam). Campuran
dibuat ulang secara dinamis, sehingga angka tersebut merupakan panjang
epoch, bukan jumlah file train yang harus disimpan di disk.

Pembagian ujaran dibuat per pembicara dengan seed 42: 80% train, 10% dev, dan
10% test. Data dinamis hanya mengambil ujaran dari bagian train.

## Pelatihan

Jalankan semua perintah dari root repositori.

### Model dua pembicara

Model dua pembicara juga menjadi sumber bobot awal untuk transfer learning.

```bash
# SkiM
python train/2speaker/skim/train_skim_2spk.py

# SkiM + Attention
python train/2speaker/skim-attention/train_skim_attention_2spk.py

# SkiM + Multi-Scale Encoder
python train/2speaker/skim-multiscale/train_skim_multiscale_2spk.py
```

Checkpoint terbaik akan disimpan di:

```text
checkpoints/2speaker/skim/best_model.pth
checkpoints/2speaker/skim-attention/best_model.pth
checkpoints/2speaker/skim-multiscale/best_model.pth
```

Path Multi-Scale di atas adalah default training manual. Wizard menambahkan
suffix kernel/stride pada nama folder untuk setiap konfigurasi.

### Model tiga pembicara dari awal

```bash
# SkiM
python train/3speaker/skim/train_skim_3spk.py

# SkiM + Attention
python train/3speaker/skim-attention/train_skim_attention_3spk.py

# SkiM + Multi-Scale Encoder
python train/3speaker/skim-multiscale/train_skim_multiscale_3spk.py
```

### Model tiga pembicara dengan transfer learning

Latih model dua pembicara yang sesuai terlebih dahulu. Skrip transfer learning
akan mengambil checkpoint dari folder `checkpoints/2speaker/`.

```bash
# Transfer SkiM 2 pembicara ke 3 pembicara
python train/3speaker/skim/train_skim_3spk_transfer.py

# Transfer SkiM + Attention 2 pembicara ke 3 pembicara
python train/3speaker/skim-attention/train_skim_attention_3spk_transfer.py

# Transfer SkiM + Multi-Scale Encoder 2 pembicara ke 3 pembicara
python train/3speaker/skim-multiscale/train_skim_multiscale_3spk_transfer.py
```

Checkpoint model tiga pembicara disimpan di:

```text
checkpoints/3speaker/skim/best_model.pth
checkpoints/3speaker/skim-attention/best_model.pth
checkpoints/3speaker/skim-multiscale/best_model.pth
checkpoints/3speaker/skim-transfer/best_model.pth
checkpoints/3speaker/skim-attention-transfer/best_model.pth
checkpoints/3speaker/skim-multiscale-transfer/best_model.pth
```

### Melanjutkan pelatihan

Skrip yang menyediakan opsi resume dapat diperiksa dengan `--help`:

```bash
python train/2speaker/skim/train_skim_2spk.py --help
python train/2speaker/skim-attention/train_skim_attention_2spk.py --help
python train/3speaker/skim/train_skim_3spk.py --help
python train/3speaker/skim-attention/train_skim_attention_3spk.py --help
python train/3speaker/skim/train_skim_3spk_transfer.py --help
python train/3speaker/skim-attention/train_skim_attention_3spk_transfer.py --help
```

Contoh:

```bash
python train/2speaker/skim/train_skim_2spk.py \
    --resume-from checkpoint_epoch_30.pth \
    --num-epochs 100
```

`--num-epochs` menyatakan total epoch tujuan, bukan jumlah epoch tambahan.

## Evaluasi

Evaluasi seluruh checkpoint lokal terhadap test set:

```bash
python eval/run_eval.py --local --audio-limit 0
```

Evaluasi model tertentu:

```bash
python eval/run_eval.py \
    --local \
    --models 3speaker-skim 3speaker-skim-attention \
    --audio-limit 10
```

Hasil disimpan dalam `eval/results/`. Nilai `--audio-limit` menentukan jumlah
contoh audio hasil pemisahan yang ikut disimpan untuk setiap model. Gunakan
nilai `0` bila hanya membutuhkan metrik.

Folder dataset dan checkpoint dapat diganti:

```bash
python eval/run_eval.py \
    --local \
    --dataset-dir /path/to/synthetic \
    --checkpoints-dir /path/to/checkpoints
```

## Inferensi

Pisahkan satu file campuran:

```bash
python inference/separate.py \
    --checkpoint checkpoints/3speaker/skim-attention-transfer/best_model.pth \
    --input contoh/mixture.wav \
    --output-dir separated
```

Pisahkan seluruh file WAV dalam satu folder:

```bash
python inference/separate.py \
    --checkpoint checkpoints/3speaker/skim-attention-transfer/best_model.pth \
    --input-dir dataset/synthetic/TITML-3spk-v2/test/mix \
    --output-dir separated \
    --device auto
```

Untuk setiap mixture, program membuat folder yang berisi `s1.wav`, `s2.wav`,
dan `s3.wav` sesuai jumlah keluaran model.

## Konfigurasi Path

Semua skrip pelatihan membaca lokasi dataset mentah dari `TSS_RAW_DIR`.
Jika variabel tersebut tidak diatur, path default-nya adalah
`dataset/raw/TTML-IDN`.

Skrip SkiM tertentu juga menyediakan override path melalui argumen CLI atau
variabel dalam `utils/paths.py`. Namun, untuk kompatibilitas penuh dengan
seluruh skrip SkiM dan SkiM Attention, gunakan struktur folder default yang
ditunjukkan dalam README ini.

## Konfigurasi Utama

Konfigurasi model standar yang digunakan skrip pelatihan:

- encoder dan decoder: 256 kanal, kernel 16, stride 8;
- separator: 4 blok, 256 unit, ukuran segmen 150;
- attention: 4 head;
- batch size: 8;
- jumlah epoch: 100;
- learning rate: 0,001;
- gradient clipping: 5,0;
- loss: SI-SNR dengan Permutation Invariant Training.

Default varian `skim-multiscale` memakai kernel encoder/decoder `(16,32,64)`
dengan stride 8, fitur dan mask terpisah per skala, serta bobot loss
`(0.8,0.1,0.1)`. Inference memakai waveform skala pendek. Ini merupakan
adaptasi SkiM yang terinspirasi TDNext; hasil verifikasi paper dan perbedaan
arsitektur tersedia dalam [review paper](implementation/skim_multiscale/PAPER_REVIEW.md).
Checkpoint multiskala lama tetap didukung untuk inference/evaluasi, tetapi
arsitektur baru perlu dilatih ulang. Wizard memakai folder konfigurasi sendiri
dan menyimpan checkpoint lama di lokasi semula. Untuk training manual,
gunakan `--checkpoint-dir` yang berbeda saat mengubah arsitektur atau ukuran.

GPU diperlukan untuk pelatihan dalam waktu yang praktis. Checkpoint dan
dataset audio tidak disertakan dalam Git karena ukuran berkasnya besar.
