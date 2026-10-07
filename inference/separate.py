"""Standalone speech separation inference.

Memisahkan audio campuran (mixture) menjadi sumber-sumber individual
menggunakan model SkiM atau SkiM-Attention yang sudah dilatih.

Konfigurasi model (arsitektur, jumlah speaker, segment_size, dll) dibaca
langsung dari checkpoint sehingga tidak ada parameter yang perlu dicocokkan
secara manual.

Contoh penggunaan:

    # Satu file
    python inference/separate.py \
        --checkpoint checkpoints/2speaker/skim-attention/best_model.pth \
        --input contoh/mixture.wav \
        --output-dir hasil/

    # Banyak file dalam satu folder
    python inference/separate.py \
        --checkpoint checkpoints/3speaker/skim/best_model.pth \
        --input-dir data/test/mix/ \
        --output-dir hasil/

Output: untuk tiap mixture `<nama>.wav` dihasilkan `<nama>/s1.wav`,
`<nama>/s2.wav`, ... sesuai jumlah speaker model.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from implementation.skim.skim_separator import SkiMSeparator
from implementation.skim_attention.skim_attention_separator import (
    SkiMAttentionSeparator,
)
from implementation.skim_multiscale.multiscale_encoder import MultiScaleConvEncoder
from implementation.skim_multiscale.skim_multiscale_separator import (
    SkiMMultiScaleSeparator,
)
from implementation.skim_multiscale.multiscale_decoder import MultiScaleConvDecoder

SAMPLE_RATE = 16000

DEFAULT_CONFIG = {
    "encoder": {"channel": 256, "kernel_size": 16, "stride": 8},
    "decoder": {"channel": 256, "kernel_size": 16, "stride": 8},
    "separator": {
        "input_dim": 256,
        "causal": False,
        "num_spk": 2,
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 4,
        "unit": 256,
        "segment_size": 150,
        "dropout": 0.1,
        "mem_type": "hc",
        "seg_overlap": False,
    },
}

_SKIM_KEYS = {
    "input_dim", "causal", "num_spk", "predict_noise", "nonlinear",
    "layer", "unit", "segment_size", "dropout", "mem_type", "seg_overlap",
}
_ATTN_KEYS = _SKIM_KEYS | {"num_heads"}
_MULTISCALE_KEYS = (_SKIM_KEYS - {"seg_overlap"}) | {"num_scales"}


def detect_arch(separator_cfg: dict, enc_cfg: dict = None) -> str:
    """Tentukan arsitektur dari konfigurasi separator dan encoder.

    - 'multiscale' jika encoder memiliki 'kernel_sizes'
    - 'attention' jika separator memiliki 'num_heads'
    - 'skim' untuk SkiM standar
    """
    if enc_cfg and "kernel_sizes" in enc_cfg:
        return "multiscale"
    return "attention" if "num_heads" in separator_cfg else "skim"


def load_model(checkpoint_path: Path, device: torch.device):
    """Bangun encoder/separator/decoder dan muat bobot dari checkpoint.

    Mengembalikan (encoder, separator, decoder, num_spk, arch).
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    config = ckpt.get("config", DEFAULT_CONFIG)
    enc_cfg = config.get("encoder", DEFAULT_CONFIG["encoder"])
    dec_cfg = config.get("decoder", DEFAULT_CONFIG["decoder"])
    sep_cfg = config.get("separator", DEFAULT_CONFIG["separator"])

    arch = detect_arch(sep_cfg, enc_cfg)
    num_spk = sep_cfg.get("num_spk", 2)

    if arch == "multiscale":
        encoder = MultiScaleConvEncoder(**enc_cfg)
        decoder = MultiScaleConvDecoder(**dec_cfg)
        kwargs = {k: v for k, v in sep_cfg.items() if k in _MULTISCALE_KEYS}
        separator = SkiMMultiScaleSeparator(**kwargs)
    elif arch == "attention":
        encoder = ConvEncoder(**enc_cfg)
        decoder = ConvDecoder(**dec_cfg)
        kwargs = {k: v for k, v in sep_cfg.items() if k in _ATTN_KEYS}
        separator = SkiMAttentionSeparator(**kwargs)
    else:
        encoder = ConvEncoder(**enc_cfg)
        decoder = ConvDecoder(**dec_cfg)
        kwargs = {k: v for k, v in sep_cfg.items() if k in _SKIM_KEYS}
        separator = SkiMSeparator(**kwargs)

    def submodule_state(prefix: str) -> dict:
        return {
            k[len(prefix) + 1:]: v
            for k, v in state_dict.items()
            if k.startswith(prefix + ".")
        }

    encoder.load_state_dict(submodule_state("encoder"))
    separator.load_state_dict(submodule_state("separator"))
    decoder.load_state_dict(submodule_state("decoder"))

    encoder = encoder.to(device).eval()
    separator = separator.to(device).eval()
    decoder = decoder.to(device).eval()

    return encoder, separator, decoder, num_spk, arch


def load_mixture(path: Path) -> np.ndarray:
    """Muat audio mixture, konversi ke mono 16 kHz float32."""
    audio, sr = sf.read(path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            y=audio, orig_sr=sr, target_sr=SAMPLE_RATE, res_type="polyphase"
        )
    return audio


@torch.no_grad()
def separate(encoder, separator, decoder, mixture: np.ndarray, device):
    """Pisahkan satu mixture menjadi daftar sumber (list of np.ndarray)."""
    mix = torch.from_numpy(mixture).unsqueeze(0).to(device)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)

    feats, flens = encoder(mix, lengths)
    masked, _, _ = separator(feats, flens)

    sources = []
    for m in masked:
        wav, _ = decoder(m, lengths)
        sources.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return sources


def peak_normalize(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    """Skala amplitudo agar puncak = `peak`, mencegah clipping saat disimpan.

    SI-SNR scale-invariant, jadi skala output model arbitrer; normalisasi
    hanya untuk kenyamanan mendengarkan.
    """
    max_val = float(np.max(np.abs(x)))
    if max_val > 0:
        return x / max_val * peak
    return x


def process_one(encoder, separator, decoder, num_spk, mix_path, out_root, device):
    """Pisahkan satu file dan simpan tiap sumber ke folder sendiri."""
    mixture = load_mixture(mix_path)
    sources = separate(encoder, separator, decoder, mixture, device)

    out_dir = out_root / mix_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(sources, start=1):
        sf.write(out_dir / f"s{i}.wav", peak_normalize(src), SAMPLE_RATE)
    return out_dir, len(sources)


def collect_inputs(args) -> list:
    """Kumpulkan daftar file mixture dari --input atau --input-dir."""
    if args.input:
        path = Path(args.input)
        if not path.is_file():
            sys.exit(f"File tidak ditemukan: {path}")
        return [path]

    in_dir = Path(args.input_dir)
    if not in_dir.is_dir():
        sys.exit(f"Folder tidak ditemukan: {in_dir}")
    files = sorted(in_dir.glob("*.wav"))
    if not files:
        sys.exit(f"Tidak ada file .wav di: {in_dir}")
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Inference speech separation (SkiM / SkiM-Attention).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path ke checkpoint model (.pth), mis. best_model.pth",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Satu file mixture .wav")
    group.add_argument("--input-dir", help="Folder berisi banyak file mixture .wav")
    parser.add_argument(
        "--output-dir", default="separated",
        help="Folder output (default: ./separated)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu"],
        help="Perangkat komputasi (default: auto)",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        sys.exit(f"Checkpoint tidak ditemukan: {ckpt_path}")

    print(f"Perangkat        : {device}")
    print(f"Memuat checkpoint: {ckpt_path}")
    encoder, separator, decoder, num_spk, arch = load_model(ckpt_path, device)
    print(f"Arsitektur       : {arch}")
    print(f"Jumlah speaker   : {num_spk}")

    inputs = collect_inputs(args)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Jumlah file      : {len(inputs)}")
    print(f"Folder output    : {out_root}\n")

    for mix_path in tqdm(inputs, desc="Memisahkan"):
        out_dir, n = process_one(
            encoder, separator, decoder, num_spk, mix_path, out_root, device
        )

    print(f"\nSelesai. Hasil disimpan di: {out_root.resolve()}")
    print(f"Tiap mixture menghasilkan {num_spk} file: s1.wav ... s{num_spk}.wav")


if __name__ == "__main__":
    main()
