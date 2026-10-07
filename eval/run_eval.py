import argparse
import gc
import json
import os
import sys
from itertools import permutations
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from implementation.skim.skim_separator import SkiMSeparator
from implementation.skim_attention.skim_attention_separator import SkiMAttentionSeparator
from implementation.skim_multiscale.multiscale_encoder import MultiScaleConvEncoder
from implementation.skim_multiscale.skim_multiscale_separator import (
    SkiMMultiScaleSeparator,
)
from implementation.skim_multiscale.multiscale_decoder import MultiScaleConvDecoder
SAMPLE_RATE = 16000
TARGET_LEN = SAMPLE_RATE * 5
EPS = 1e-08
EVAL_DIR = Path(__file__).resolve().parent
MODEL_LIST = EVAL_DIR / 'best-model-list.txt'
CKPT_CACHE = EVAL_DIR / 'ckpts'
RESULTS_DIR = EVAL_DIR / 'results'
AUDIO_LIMIT_DEFAULT = 450
DEFAULT_DATASET_DIR = ROOT / 'dataset' / 'synthetic'
DEFAULT_CHECKPOINTS_DIR = ROOT / 'checkpoints'
CHECKPOINTS_DIR = DEFAULT_CHECKPOINTS_DIR
TEST_ROOT = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_CONFIG = {'encoder': {'channel': 256, 'kernel_size': 16, 'stride': 8}, 'decoder': {'channel': 256, 'kernel_size': 16, 'stride': 8}, 'separator': {'input_dim': 256, 'causal': False, 'layer': 4, 'unit': 256, 'segment_size': 150, 'dropout': 0.1, 'mem_type': 'hc', 'seg_overlap': False, 'nonlinear': 'relu'}}

_SKIM_KEYS = {'input_dim', 'causal', 'num_spk', 'predict_noise', 'nonlinear', 'layer', 'unit', 'segment_size', 'dropout', 'mem_type', 'seg_overlap'}
_ATTN_KEYS = _SKIM_KEYS | {'num_heads'}
_MULTISCALE_KEYS = (_SKIM_KEYS - {'seg_overlap'}) | {'num_scales'}

def build_config(num_spk: int, arch: str) -> dict:
    cfg = {'encoder': dict(DEFAULT_CONFIG['encoder']), 'decoder': dict(DEFAULT_CONFIG['decoder']), 'separator': dict(DEFAULT_CONFIG['separator'])}
    cfg['separator']['num_spk'] = num_spk
    if arch == 'attention':
        cfg['separator']['num_heads'] = 4
    elif arch == 'multiscale':
        cfg['encoder'] = {'channel': 256, 'out_channel': 256, 'kernel_sizes': (16, 32, 64), 'stride': 8, 'causal': False}
    return cfg

def parse_model_name(name: str) -> tuple[int, str]:
    if name.startswith('2speaker'):
        num_spk = 2
    elif name.startswith('3speaker'):
        num_spk = 3
    else:
        raise ValueError(f'cannot parse num_spk from {name}')
    if 'skim-multiscale' in name:
        arch = 'multiscale'
    elif 'skim-attention' in name:
        arch = 'attention'
    else:
        arch = 'skim'
    return (num_spk, arch)

def read_model_list() -> list[tuple[str, str]]:
    if not MODEL_LIST.exists():
        return []
    out = []
    for line in MODEL_LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, gid = line.split('=', 1)
        out.append((name.strip(), gid.strip()))
    return out

def local_ckpt_path(name: str) -> Path:
    """Petakan nama model -> path checkpoint lokal.

    '2speaker-skim-attention' -> checkpoints/2speaker/skim-attention/best_model.pth
    """
    spk_token, _, rest = name.partition('-')
    return CHECKPOINTS_DIR / spk_token / rest / 'best_model.pth'

def discover_local_models() -> list[tuple[str, str]]:
    """Temukan semua checkpoint lokal di checkpoints/<spk>/<arch>/best_model.pth.

    gdrive_id dikosongkan karena tidak perlu download.
    """
    out = []
    for ckpt in sorted(CHECKPOINTS_DIR.glob('*/*/best_model.pth')):
        spk = ckpt.parent.parent.name
        arch = ckpt.parent.name
        out.append((f'{spk}-{arch}', ''))
    return out

def ensure_checkpoint(name: str, gdrive_id: str) -> Path:
    local = local_ckpt_path(name)
    if local.exists():
        return local
    target = CKPT_CACHE / name / 'best_model.pth'
    if target.exists():
        return target
    if not gdrive_id:
        raise FileNotFoundError(f'checkpoint lokal tidak ditemukan untuk {name} (dicari di {local}) dan tidak ada gdrive_id')
    target.parent.mkdir(parents=True, exist_ok=True)
    import gdown
    url = f'https://drive.google.com/uc?id={gdrive_id}'
    print(f'[download] {name} ← {url}')
    gdown.download(url, str(target), quiet=False)
    if not target.exists():
        raise RuntimeError(f'download failed for {name}')
    return target

def load_wav(p: Path) -> np.ndarray:
    a, sr = sf.read(p)
    assert sr == SAMPLE_RATE, f'{p}: sr={sr}'
    a = a.astype(np.float32)
    if len(a) > TARGET_LEN:
        return a[:TARGET_LEN]
    if len(a) < TARGET_LEN:
        return np.pad(a, (0, TARGET_LEN - len(a)))
    return a

def si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + EPS) * ref
    e_noise = est - s_target
    return float(10 * np.log10((np.dot(s_target, s_target) + EPS) / (np.dot(e_noise, e_noise) + EPS)))

def best_pit(ests, refs) -> tuple[float, tuple[int, ...]]:
    best_val = -1000000000.0
    best_perm = None
    for perm in permutations(range(len(refs))):
        v = float(np.mean([si_snr(ests[perm[i]], refs[i]) for i in range(len(refs))]))
        if v > best_val:
            best_val = v
            best_perm = perm
    return (best_val, best_perm)

def build_model(num_spk: int, arch: str, ckpt_path: Path):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get('model_state_dict', state)
    cfg = state.get('config') or build_config(num_spk, arch)
    sep_cfg = dict(cfg['separator'])
    sep_cfg.setdefault('num_spk', num_spk)
    if arch == 'multiscale':
        enc = MultiScaleConvEncoder(**cfg['encoder'])
        dec = MultiScaleConvDecoder(**cfg['decoder'])
        sep = SkiMMultiScaleSeparator(**{k: v for k, v in sep_cfg.items() if k in _MULTISCALE_KEYS})
    elif arch == 'skim':
        enc = ConvEncoder(**cfg['encoder'])
        dec = ConvDecoder(**cfg['decoder'])
        sep = SkiMSeparator(**{k: v for k, v in sep_cfg.items() if k in _SKIM_KEYS})
    elif arch == 'attention':
        enc = ConvEncoder(**cfg['encoder'])
        dec = ConvDecoder(**cfg['decoder'])
        sep = SkiMAttentionSeparator(**{k: v for k, v in sep_cfg.items() if k in _ATTN_KEYS})
    else:
        raise ValueError(arch)
    enc, sep, dec = (enc.to(device).eval(), sep.to(device).eval(), dec.to(device).eval())

    def split_sd(prefix):
        return {k.replace(f'{prefix}.', '', 1): v for k, v in sd.items() if k.startswith(f'{prefix}.')}
    enc.load_state_dict(split_sd('encoder'))
    sep.load_state_dict(split_sd('separator'))
    dec.load_state_dict(split_sd('decoder'))
    return (enc, sep, dec)

@torch.no_grad()
def separate(enc, sep, dec, mix_np: np.ndarray) -> list[np.ndarray]:
    mix = torch.from_numpy(mix_np).unsqueeze(0).to(device)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)
    feats, flens = enc(mix, lengths)
    masked, _, _ = sep(feats, flens)
    out = []
    for m in masked:
        wav, _ = dec(m, lengths)
        out.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return out

def save_audio_set(out_dir: Path, fid: str, mix_np, refs, ests, perm):
    d = out_dir / fid
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / 'mixture.wav', mix_np, SAMPLE_RATE)
    for i, ref in enumerate(refs, start=1):
        sf.write(d / f's{i}_gt.wav', ref, SAMPLE_RATE)
    for i in range(len(refs)):
        est_aligned = ests[perm[i]]
        if len(est_aligned) > TARGET_LEN:
            est_aligned = est_aligned[:TARGET_LEN]
        elif len(est_aligned) < TARGET_LEN:
            est_aligned = np.pad(est_aligned, (0, TARGET_LEN - len(est_aligned)))
        sf.write(d / f's{i + 1}_est.wav', est_aligned, SAMPLE_RATE)

def normalize_len(x):
    if len(x) > TARGET_LEN:
        return x[:TARGET_LEN]
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x

def eval_model(name: str, gdrive_id: str, audio_limit: int) -> dict:
    num_spk, arch = parse_model_name(name)
    test_root = TEST_ROOT[num_spk]
    if not (test_root / 'mix').exists():
        raise FileNotFoundError(f'test dataset not found: {test_root}')
    ckpt_path = ensure_checkpoint(name, gdrive_id)
    enc, sep, dec = build_model(num_spk, arch, ckpt_path)
    out_dir = RESULTS_DIR / name
    audio_dir = out_dir / 'audio'
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    mix_files = sorted((test_root / 'mix').glob('*.wav'))
    print(f'\n=== {name} (num_spk={num_spk}, arch={arch}, test={len(mix_files)}) ===')
    csv_path = out_dir / 'per_file.csv'
    csv_f = csv_path.open('w', buffering=1)
    csv_f.write('file_id,sisnr,perm\n')
    sisnrs = []
    for idx, mp in enumerate(tqdm(mix_files, desc=name)):
        fid = mp.stem
        mix_np = load_wav(mp)
        refs = [load_wav(test_root / f's{i}' / f'{fid}.wav') for i in range(1, num_spk + 1)]
        ests = [normalize_len(e) for e in separate(enc, sep, dec, mix_np)]
        sep_si, perm = best_pit(ests, refs)
        sisnrs.append(sep_si)
        csv_f.write(f"{fid},{sep_si:.4f},{'-'.join(map(str, perm))}\n")
        if idx < audio_limit:
            save_audio_set(audio_dir, fid, mix_np, refs, ests, perm)
    csv_f.close()
    arr_si = np.array(sisnrs)
    stats = {'model': name, 'num_spk': num_spk, 'arch': arch, 'n_eval': int(len(arr_si)), 'n_audio_saved': int(min(audio_limit, len(mix_files))), 'mean_sisnr': float(arr_si.mean()), 'std_sisnr': float(arr_si.std()), 'median_sisnr': float(np.median(arr_si))}
    (out_dir / 'stats.json').write_text(json.dumps(stats, indent=2))
    print(f"  mean SI-SNR : {stats['mean_sisnr']:.3f} ± {stats['std_sisnr']:.3f} dB")
    del enc, sep, dec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return stats

def main():
    global CHECKPOINTS_DIR, TEST_ROOT
    p = argparse.ArgumentParser(description='Evaluasi SI-SNR model SkiM / SkiM-Attention pada test set.')
    p.add_argument('--models', nargs='*', default=None, help='subset nama model yang dievaluasi')
    p.add_argument('--dataset-dir', default=str(DEFAULT_DATASET_DIR), help='folder berisi TITML-2spk-v2/ dan TITML-3spk-v2/ (default: ROOT/dataset/synthetic)')
    p.add_argument('--checkpoints-dir', default=str(DEFAULT_CHECKPOINTS_DIR), help='folder checkpoints/<spk>/<arch>/best_model.pth (default: ROOT/checkpoints)')
    p.add_argument('--local', action='store_true', help='temukan otomatis semua checkpoint lokal di --checkpoints-dir (tanpa perlu best-model-list.txt)')
    p.add_argument('--audio-limit', type=int, default=int(os.environ.get('AUDIO_LIMIT', AUDIO_LIMIT_DEFAULT)))
    args = p.parse_args()

    CHECKPOINTS_DIR = Path(args.checkpoints_dir).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    TEST_ROOT = {2: dataset_dir / 'TITML-2spk-v2' / 'test', 3: dataset_dir / 'TITML-3spk-v2' / 'test'}

    print(f'device         : {device}')
    print(f'dataset_dir    : {dataset_dir}')
    print(f'checkpoints_dir: {CHECKPOINTS_DIR}')
    print(f'audio_limit    : {args.audio_limit}')
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_CACHE.mkdir(parents=True, exist_ok=True)

    entries = discover_local_models() if args.local else read_model_list()
    if not entries:
        sys.exit('tidak ada model untuk dievaluasi. Pakai --local untuk memindai checkpoint lokal, atau sediakan best-model-list.txt')
    if args.models:
        wanted = set(args.models)
        entries = [(n, g) for n, g in entries if n in wanted]
        if not entries:
            sys.exit(f'no matching models in list: {args.models}')
    summary = {}
    summary_path = RESULTS_DIR / 'summary.json'
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}
    for name, gid in entries:
        try:
            stats = eval_model(name, gid, args.audio_limit)
            summary[name] = stats
            summary_path.write_text(json.dumps(summary, indent=2))
        except Exception as e:
            print(f'[FAIL] {name}: {e}')
            summary[name] = {'error': str(e)}
            summary_path.write_text(json.dumps(summary, indent=2))
    print('\n=== FINAL SUMMARY ===')
    print(json.dumps(summary, indent=2))
if __name__ == '__main__':
    main()
