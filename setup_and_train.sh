#!/usr/bin/env bash
# Interactive setup and training for the clean TA-speech-separation repository.
set -euo pipefail

REPO_URL=https://github.com/aqiilaah/TA-speech-separation.git
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONFIG_FILE="$SCRIPT_DIR/setup_and_train.env"
CONFIGURE=0
case "${1:-}" in
    --configure) CONFIGURE=1 ;;
    --help|-h)
        printf '%s\n' \
            'Usage: bash setup_and_train.sh [--configure]' \
            'First run asks for paths, model, epochs, total dataset hours, and ZIP source.' \
            'Python 3.10/3.11 is detected automatically; missing Python 3.11 is installed with uv.' \
            'Later runs reuse setup_and_train.env and resume completed checkpoints.' \
            'Use --configure to change saved choices. ZIP passwords are not saved.' \
            'Requires Bash, Git, curl for automatic Python installation, and an NVIDIA GPU for CUDA mode.'
        exit 0 ;;
    '') ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
stage() { printf '\n[%s/7] %s\n' "$1" "$2"; }
ask() {
    local variable=$1 question=$2 default=$3 reply
    printf '%s [%s]: ' "$question" "$default"
    IFS= read -r reply || die 'Input closed. Run script in an interactive terminal.'
    printf -v "$variable" '%s' "${reply:-$default}"
}

python_is_compatible() {
    [[ -n "$1" ]] && command -v "$1" >/dev/null 2>&1 &&
        "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 11) else 1)' \
        >/dev/null 2>&1
}

ensure_python() {
    local candidate uv_bin
    for candidate in "${PYTHON_BIN:-}" "$PROJECT_DIR/.venv/bin/python" \
        python3.11 python3.10 python3 python; do
        if python_is_compatible "$candidate"; then
            PYTHON_BIN=$(command -v "$candidate")
            printf 'Using compatible Python: %s\n' "$PYTHON_BIN"
            return
        fi
    done

    printf 'No compatible Python found; installing Python 3.11 alongside system Python.\n'
    uv_bin=$(command -v uv || true)
    if [[ -z "$uv_bin" ]]; then
        uv_bin="$HOME/.local/bin/uv"
        if [[ ! -x "$uv_bin" ]]; then
            command -v curl >/dev/null 2>&1 || die 'Install curl to bootstrap Python automatically.'
            printf 'Installing uv in %s (shell profiles unchanged).\n' "$HOME/.local/bin"
            if ! curl -LsSf --retry 3 https://astral.sh/uv/install.sh |
                env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh; then
                die 'Could not install uv. Check network access and rerun.'
            fi
        fi
    fi
    "$uv_bin" python install 3.11 || die 'Could not install Python 3.11. Check network access and rerun.'
    PYTHON_BIN=$("$uv_bin" python find --managed-python 3.11) || die 'Could not locate installed Python 3.11.'
    python_is_compatible "$PYTHON_BIN" || die 'Installed Python lacks required venv/ensurepip support.'
    printf 'Using installed Python: %s\n' "$PYTHON_BIN"
}

if [[ -f "$CONFIG_FILE" && "$CONFIGURE" == 0 ]]; then
    # This file is generated below with shell-escaped values and mode 600.
    # shellcheck disable=SC1090
    source "$CONFIG_FILE"
    printf 'Using saved choices: %s\n' "$CONFIG_FILE"
else
    printf '\nTA-speech-separation: first-run setup\n'
    default_project="$PWD/TA-speech-separation"
    if [[ -f "$SCRIPT_DIR/train/datasets_utils.py" ]]; then
        default_project="$SCRIPT_DIR"
    fi
    ask PROJECT_DIR 'Project folder (clone here, or reuse existing clean checkout)' \
        "$default_project"
    ask DEVICE 'PyTorch build: cuda or cpu' cuda
    ask GPU_ID 'CUDA GPU index (ignored for cpu)' 0
    printf '\nChoose model:\n'
    printf '%s\n' \
        '  1. SkiM, 2 speakers' \
        '  2. SkiM + Attention, 2 speakers' \
        '  3. SkiM, 3 speakers' \
        '  4. SkiM + Attention, 3 speakers' \
        '  5. SkiM transfer, 2 speakers to 3 speakers' \
        '  6. SkiM + Attention transfer, 2 speakers to 3 speakers' \
        '  7. SkiM + Multi-Scale, 2 speakers' \
        '  8. SkiM + Multi-Scale, 3 speakers' \
        '  9. SkiM + Multi-Scale transfer, 2 speakers to 3 speakers'
    ask MODEL 'Model number' 2
    ask EPOCHS 'Total training epochs (1 for an initial trial)' 100
    PRETRAIN_EPOCHS=100
    if [[ "$MODEL" == 5 || "$MODEL" == 6 || "$MODEL" == 9 ]]; then
        ask PRETRAIN_EPOCHS 'Two-speaker pretraining epochs if source is missing' 100
    fi
    printf '\nHours refer to mixtures, not raw recordings or all epochs combined.\n'
    printf 'Split: 80%% dynamic train per epoch, 10%% validation, 10%% test.\n'
    ask TARGET_HOURS 'Total dataset hours (10 = 8 train + 1 validation + 1 test)' 10
    ask ZIP_SOURCE 'TITML ZIP URL or local ZIP path' \
        'https://drive.google.com/uc?id=1ETEzZhm5s5XAp1tKUtSj9zq4Ic-7tben'
fi

[[ "$DEVICE" == cuda || "$DEVICE" == cpu ]] || die 'Device must be cuda or cpu.'
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || die 'GPU index must be a nonnegative integer.'
[[ "$MODEL" =~ ^[1-9]$ ]] || die 'Choose model 1 through 9.'
for value in "$EPOCHS" "$PRETRAIN_EPOCHS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die 'Epoch counts must be positive integers.'
done
[[ "$TARGET_HOURS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die 'Dataset hours must be a positive number.'
case "$MODEL" in
    1|3|5) variant=skim ;;
    2|4|6) variant=skim-attention ;;
    7|8|9) variant=skim-multiscale ;;
esac
stem=${variant//-/_}
speakers=2
[[ "$MODEL" == 1 || "$MODEL" == 2 || "$MODEL" == 7 ]] || speakers=3
training_script="train/${speakers}speaker/$variant/train_${stem}_${speakers}spk.py"
training_scripts=("$training_script")
if [[ "$MODEL" == 5 || "$MODEL" == 6 || "$MODEL" == 9 ]]; then
    pretrain_script="train/2speaker/$variant/train_${stem}_2spk.py"
    training_script="train/3speaker/$variant/train_${stem}_3spk_transfer.py"
    training_scripts=("$pretrain_script" "$training_script")
fi
PROJECT_DIR=${PROJECT_DIR/#\~/$HOME}
ensure_python
PYTHON_BIN=$("$PYTHON_BIN" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$PYTHON_BIN")
"$PYTHON_BIN" -c 'import math,sys; hours=float(sys.argv[1]); assert math.isfinite(hours) and hours >= 1, "Use at least 1 finite dataset hour"' "$TARGET_HOURS"
TRAIN_EPOCH_SIZE=$("$PYTHON_BIN" -c 'import sys; total=int(float(sys.argv[1])*3600/5); print(int(total*0.8))' "$TARGET_HOURS")
PROJECT_DIR=$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$PROJECT_DIR")
ZIP_SOURCE=${ZIP_SOURCE/#\~/$HOME}
if [[ -f "$ZIP_SOURCE" ]]; then
    ZIP_SOURCE=$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$ZIP_SOURCE")
fi

umask 077
for variable in PROJECT_DIR PYTHON_BIN DEVICE GPU_ID MODEL EPOCHS PRETRAIN_EPOCHS TARGET_HOURS ZIP_SOURCE; do
    printf '%s=%q\n' "$variable" "${!variable}"
done > "$CONFIG_FILE.tmp"
mv -- "$CONFIG_FILE.tmp" "$CONFIG_FILE"

stage 1 'Clone clean project'
command -v git >/dev/null 2>&1 || die 'Install Git first.'
if [[ -d "$PROJECT_DIR/.git" ]]; then
    remote=$(git -C "$PROJECT_DIR" config --get remote.origin.url)
    case "$remote" in
        https://github.com/aqiilaah/TA-speech-separation|\
        https://github.com/aqiilaah/TA-speech-separation.git|\
        git@github.com:aqiilaah/TA-speech-separation.git|\
        https://github.com/Fadil-Tao/TA-speech-separation|\
        https://github.com/Fadil-Tao/TA-speech-separation.git|\
        git@github.com:Fadil-Tao/TA-speech-separation.git) ;;
        *) die "Existing directory is not the clean repo: $remote" ;;
    esac
    printf 'Reusing clean checkout: %s\n' "$PROJECT_DIR"
else
    git clone --branch main "$REPO_URL" "$PROJECT_DIR"
fi
cd -- "$PROJECT_DIR"

# Check required scripts explicitly: globbing alone misses absent model files.
"$PYTHON_BIN" - "$REPO_URL" "${training_scripts[@]}" <<'PY'
import ast
import sys
from pathlib import Path

for filename in sys.argv[2:]:
    path = Path(filename)
    if not path.is_file():
        raise SystemExit(
            f'Checkout is missing required training script: {path}\n'
            f'Project folder: {Path.cwd()}\n'
            f'Use an updated checkout of {sys.argv[1]} and rerun --configure '
            'to select it. Existing datasets and checkpoints can be kept.'
        )
    for option in ('--num-epochs', '--resume-from'):
        if option not in path.read_text():
            raise SystemExit(f'{path} lacks {option}. Update checkout first.')

dataset = Path('train/datasets_utils.py').read_text()
if 'class DynamicMixDataset' not in dataset:
    raise SystemExit(
        'Clone lacks DynamicMixDataset. Publish pending clean-repo updates '
        'first, or rerun --configure and choose the updated local checkout.'
    )
for path in Path('train').glob('*speaker/*/train*.py'):
    tree = ast.parse(path.read_text())
    dynamic = any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'train_dataset'
                for t in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'DynamicMixDataset'
        for node in ast.walk(tree)
    )
    if not dynamic:
        raise SystemExit(f'{path} still uses static training. Update clone first.')
    if 'TSS_TRAIN_EPOCH_SIZE' not in path.read_text():
        raise SystemExit(f'{path} lacks configurable train size. Update clone first.')
for speakers in (2, 3):
    path = Path(f'dataset/generator/titml_mix_generator_{speakers}spk.py')
    if '--only-splits' not in path.read_text():
        raise SystemExit(f'{path} lacks --only-splits. Update clone first.')
PY

stage 2 'Set up virtual environment and dependencies'
if [[ -e .venv || -L .venv ]] && ! python_is_compatible .venv/bin/python; then
    backup=$(mktemp -d "$PROJECT_DIR/.venv.backup.XXXXXX")
    mv -- .venv "$backup/original"
    printf 'Preserved incompatible virtual environment: %s/original\n' "$backup"
fi
if python_is_compatible .venv/bin/python; then
    printf 'Reusing compatible virtual environment; creation skipped.\n'
else
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python_is_compatible python || die 'Virtual environment failed Python compatibility check.'
python -c 'import pip' >/dev/null 2>&1 || python -m ensurepip --upgrade
python -m pip install --upgrade pip wheel
index=https://download.pytorch.org/whl/cu124
[[ "$DEVICE" == cuda ]] || index=https://download.pytorch.org/whl/cpu
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url "$index"
python -m pip install 'numpy<1.24' espnet==202304 espnet_model_zoo \
    librosa==0.9.2 soundfile matplotlib tqdm mir_eval pesq pystoi 'gdown>=6,<7' pyzipper
if [[ "$DEVICE" == cuda ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; check NVIDIA driver/GPU or reconfigure for CPU"; print("GPU:", torch.cuda.get_device_name(0))'
else
    export CUDA_VISIBLE_DEVICES=''
    printf 'CPU training selected; full training will be slow.\n'
fi
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export TSS_PROJECT_ROOT="$PROJECT_DIR"
export TSS_RAW_DIR="$PROJECT_DIR/dataset/raw/TTML-IDN"
export TSS_SYNTHETIC_DIR="$PROJECT_DIR/dataset/synthetic"
export TSS_CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
export TSS_TRAIN_EPOCH_SIZE="$TRAIN_EPOCH_SIZE"

stage 3 'Download TITML source ZIP'
ZIP_PATH="$PROJECT_DIR/dataset/zips/TITML-IDN.zip"
if [[ -d "$TSS_RAW_DIR/Speech" ]]; then
    printf 'Raw Speech/ already exists; download skipped.\n'
elif [[ -f "$ZIP_SOURCE" ]]; then
    ZIP_PATH="$ZIP_SOURCE"
    printf 'Using existing ZIP: %s\n' "$ZIP_PATH"
elif [[ -s "$ZIP_PATH" ]]; then
    printf 'Using cached ZIP: %s\n' "$ZIP_PATH"
else
    mkdir -p -- "$(dirname -- "$ZIP_PATH")"
    case "$ZIP_SOURCE" in
        *drive.google.com/*)
            python -m gdown "$ZIP_SOURCE" -O "$ZIP_PATH.partial" ;;
        https://*|http://*)
            command -v curl >/dev/null 2>&1 || die 'Install curl for direct ZIP URLs.'
            curl --fail --location --retry 3 "$ZIP_SOURCE" -o "$ZIP_PATH.partial" ;;
        *) die 'ZIP source must be an existing file or an HTTP(S) URL.' ;;
    esac
    mv -- "$ZIP_PATH.partial" "$ZIP_PATH"
fi

stage 4 'Extract raw dataset'
if [[ ! -d "$TSS_RAW_DIR/Speech" ]]; then
    encrypted=$(python - "$ZIP_PATH" <<'PY'
import sys
import pyzipper
with pyzipper.AESZipFile(sys.argv[1]) as archive:
    print(int(any(item.flag_bits & 1 for item in archive.infolist())))
PY
    )
    if [[ "$encrypted" == 1 && -z "${TSS_ZIP_PASSWORD:-}" ]]; then
        # Reuse the parent's known password when running from this workspace.
        parent_setup="$SCRIPT_DIR/../scripts/setup_datasets.sh"
        if [[ -f "$parent_setup" ]]; then
            while IFS= read -r line; do
                if [[ "$line" =~ ^RAW_PASSWORD=\"([^\"]+)\"$ ]]; then
                    TSS_ZIP_PASSWORD=${BASH_REMATCH[1]}
                    break
                fi
            done < "$parent_setup"
        fi
        if [[ -z "${TSS_ZIP_PASSWORD:-}" ]]; then
            IFS= read -r -s -p 'Encrypted ZIP password: ' TSS_ZIP_PASSWORD
            printf '\n'
        fi
    fi
    export TSS_ZIP_PASSWORD=${TSS_ZIP_PASSWORD:-}
    python - "$ZIP_PATH" "$TSS_RAW_DIR" <<'PY'
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pyzipper

destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
    root = Path(temporary).resolve()
    with pyzipper.AESZipFile(sys.argv[1]) as archive:
        password = os.environ.get('TSS_ZIP_PASSWORD', '')
        if password:
            archive.setpassword(password.encode())
        for item in archive.infolist():
            target = (root / item.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe ZIP entry: {item.filename}')
        archive.extractall(root)
    speech_dirs = [path for path in root.rglob('Speech') if path.is_dir()]
    if len(speech_dirs) != 1:
        raise ValueError('ZIP must contain exactly one Speech/ directory.')
    if not any(speech_dirs[0].glob('*/*.wav')):
        raise ValueError('Speech/ contains no speaker WAV files.')
    shutil.move(str(speech_dirs[0]), str(destination / 'Speech'))
print(f'Extracted: {destination / "Speech"}')
PY
    unset TSS_ZIP_PASSWORD
else
    printf 'Raw Speech/ already exists; extraction skipped.\n'
fi

stage 5 'Generate static validation (dev) and test mixtures only'
printf 'Total mixture-hours: %s (80%% train / 10%% validation / 10%% test).\n' "$TARGET_HOURS"
printf 'Dynamic train: %s five-second mixtures per epoch.\n' "$TSS_TRAIN_EPOCH_SIZE"
validate_dataset() {
    python - "$1" "$2" "$TARGET_HOURS" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
speakers = int(sys.argv[2])
total = int(float(sys.argv[3]) * 3600 / 5)
train_count, dev_count = int(total * 0.8), int(total * 0.1)
expected = {'dev': dev_count, 'test': total - train_count - dev_count}
for split, count in expected.items():
    ids = {p.stem for p in (root / split / 'mix').glob('*.wav')}
    if len(ids) != count:
        raise SystemExit(f'{split}: expected {count} mixtures, found {len(ids)}')
    for source in range(1, speakers + 1):
        source_ids = {p.stem for p in (root / split / f's{source}').glob('*.wav')}
        if source_ids != ids:
            raise SystemExit(f'{split}/s{source}: missing or mismatched sources')
PY
}
counts=("$speakers")
if [[ "$MODEL" == 5 || "$MODEL" == 6 || "$MODEL" == 9 ]]; then
    counts=(2 3)
fi
for count in "${counts[@]}"; do
    dataset_dir="$TSS_SYNTHETIC_DIR/TITML-${count}spk-v2"
    marker="$dataset_dir/.wizard-dev-test-${TARGET_HOURS}h-seed42"
    preparing="$dataset_dir/.wizard-preparing-${TARGET_HOURS}h-seed42"
    if [[ ! -f "$marker" ]]; then
        if [[ ! -f "$preparing" && ( -d "$dataset_dir/dev" || -d "$dataset_dir/test" ) ]]; then
            die "Existing $dataset_dir was not prepared for ${TARGET_HOURS}h by this wizard. Choose a fresh project folder to preserve existing data and checkpoints."
        fi
        mkdir -p -- "$dataset_dir"
        touch -- "$preparing"
        python "dataset/generator/titml_mix_generator_${count}spk.py" \
            --titml-dir "$TSS_RAW_DIR" --output-dir "$dataset_dir" \
            --target-hours "$TARGET_HOURS" --target-duration 5 \
            --train-ratio 0.8 --dev-ratio 0.1 --seed 42 \
            --only-splits dev test
        validate_dataset "$dataset_dir" "$count"
        touch -- "$marker"
    else
        validate_dataset "$dataset_dir" "$count"
        printf 'Reusing completed dev/test: %s\n' "$dataset_dir"
    fi
done

stage 6 'Train model with on-the-fly mixtures'
run_training() {
    local script=$1 checkpoint_dir=$2 epochs=$3 resume
    local -a command
    mkdir -p -- "$checkpoint_dir"
    if python - "$checkpoint_dir" "$epochs" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
history = root / 'training_history.json'
if not history.exists() or not (root / 'best_model.pth').exists():
    sys.exit(1)
data = json.loads(history.read_text())
sys.exit(0 if len(data.get('val_losses', [])) >= int(sys.argv[2]) else 1)
PY
    then
        printf 'Training already completed: %s\n' "$checkpoint_dir"
        return
    fi
    resume=$(python - "$checkpoint_dir" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
paths = list(root.glob('checkpoint_epoch_*.pth'))
if (root / 'best_model.pth').exists():
    paths.append(root / 'best_model.pth')
checkpoints = [
    (int(torch.load(path, map_location='cpu', weights_only=False)['epoch']), path)
    for path in paths
]
if checkpoints:
    print(max(checkpoints, key=lambda item: item[0])[1])
PY
    )
    command=(python -u "$script" --num-epochs "$epochs")
    if [[ -n "$resume" ]]; then
        printf 'Resuming completed checkpoint: %s\n' "$resume"
        command+=(--resume-from "$resume")
    fi
    "${command[@]}" 2>&1 | tee -a "$checkpoint_dir/run.log"
    python - "$checkpoint_dir/training_history.json" "$epochs" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
history = json.loads(path.read_text()) if path.exists() else {}
if len(history.get('val_losses', [])) < int(sys.argv[2]):
    raise SystemExit('Training stopped before requested epochs. Rerun to resume.')
PY
}

if [[ "$MODEL" == 5 || "$MODEL" == 6 || "$MODEL" == 9 ]]; then
    source_dir="$TSS_CHECKPOINT_DIR/2speaker/$variant"
    if [[ ! -f "$source_dir/best_model.pth" ]]; then
        printf 'Transfer source missing; training %s with 2 speakers first.\n' "$variant"
        run_training "$pretrain_script" "$source_dir" "$PRETRAIN_EPOCHS"
    fi
    # Avoid an inherited override redirecting transfer to a different checkpoint.
    export TSS_PRETRAINED_PATH="$source_dir/best_model.pth"
    output_dir="$TSS_CHECKPOINT_DIR/3speaker/${variant}-transfer"
    run_training "$training_script" "$output_dir" "$EPOCHS"
else
    output_dir="$TSS_CHECKPOINT_DIR/${speakers}speaker/$variant"
    run_training "$training_script" "$output_dir" "$EPOCHS"
fi

stage 7 'Finished'
printf 'Best model: %s/best_model.pth\n' "$output_dir"
printf 'Training log: %s/run.log\n' "$output_dir"
printf 'Saved choices: %s\n' "$CONFIG_FILE"
