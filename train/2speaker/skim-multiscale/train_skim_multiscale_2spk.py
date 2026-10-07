import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import site

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train"))

user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

from datasets_utils import (
    DynamicMixDataset,
    IndonesianMixDataset,
    build_utterance_split,
)
from utils.paths import get_raw_dir, get_synthetic_dir, get_checkpoint_dir
from utils.multiscale_config import apply_multiscale_environment

try:
    from implementation.skim_multiscale.multiscale_model import (
        MultiScaleEnhancementModel,
        MultiScalePITSolver,
    )
    from espnet2.enh.loss.criterions.time_domain import SISNRLoss
except ImportError:
    MultiScaleEnhancementModel = None
    SISNRLoss = None
    MultiScalePITSolver = None

from implementation.skim_multiscale.multiscale_encoder import MultiScaleConvEncoder
from implementation.skim_multiscale.skim_multiscale_separator import (
    SkiMMultiScaleSeparator,
)
from implementation.skim_multiscale.multiscale_decoder import MultiScaleConvDecoder

MODEL_CONFIG = {
    "encoder": {
        "channel": 256,
        "kernel_sizes": (16, 32, 64),
        "stride": 8,
        "causal": False,
        "nonlinear": "relu",
        "preserve_scales": True,
    },
    "decoder": {
        "channel": 256,
        "kernel_sizes": (16, 32, 64),
        "stride": 8,
        "causal": False,
    },
    "separator": {
        "input_dim": 256,
        "num_scales": 3,
        "causal": False,
        "num_spk": 2,
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 4,
        "unit": 256,
        "segment_size": 150,
        "dropout": 0.1,
        "mem_type": "hc",
    },
}

apply_multiscale_environment(MODEL_CONFIG)

TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "gradient_clip": 5.0,
    "scale_loss_weights": (0.8, 0.1, 0.1),
    "seed": 42,
}

DATASET_DIR = get_synthetic_dir("TITML-2spk-v2")
RAW_DIR = get_raw_dir()
CHECKPOINT_DIR = get_checkpoint_dir("2speaker", "skim-multiscale")


def build_model(device):
    if MultiScaleEnhancementModel is None:
        raise ImportError(
            "ESPnet2 tidak ditemukan di environment Python saat ini. "
            "Pastikan telah menginstal dependensi: pip install espnet espnet_model_zoo"
        )

    print("\n" + "=" * 60)
    print("Building SkiM Multi-Scale 2-Speaker Model")
    print("=" * 60)

    encoder = MultiScaleConvEncoder(**MODEL_CONFIG["encoder"])
    print(
        f"Encoder: MultiScaleConvEncoder (kernels: {MODEL_CONFIG['encoder']['kernel_sizes']}, "
        f"stride: {MODEL_CONFIG['encoder']['stride']}, out: {encoder.output_dim})"
    )

    separator = SkiMMultiScaleSeparator(**MODEL_CONFIG["separator"])
    print(
        f"Separator: SkiMMultiScaleSeparator ({MODEL_CONFIG['separator']['layer']} layers, "
        f"{MODEL_CONFIG['separator']['unit']} units, {MODEL_CONFIG['separator']['num_spk']} speakers)"
    )

    decoder = MultiScaleConvDecoder(**MODEL_CONFIG["decoder"])
    print(
        f"Decoder: MultiScaleConvDecoder (kernels: {MODEL_CONFIG['decoder']['kernel_sizes']}, "
        f"stride: {MODEL_CONFIG['decoder']['stride']})"
    )

    criterion = SISNRLoss()
    pit_wrapper = MultiScalePITSolver(criterion, TRAIN_CONFIG["scale_loss_weights"])
    print(f"Loss: multi-scale SI-SNR with shared PIT, weights {pit_wrapper.scale_weights}")

    model = MultiScaleEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[pit_wrapper],
        loss_type="si_snr",
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    return model


def save_training_history(train_losses, val_losses, out_dir):
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)


def load_training_history(out_dir):
    history_path = out_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
        return data.get("train_losses", []), data.get("val_losses", [])
    return [], []


def save_training_curves(train_losses, val_losses, out_dir):
    if not train_losses:
        return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses, label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative SI-SNR)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label="Train SI-SNR", marker="o", markersize=3)
    plt.plot([-l for l in val_losses], label="Val SI-SNR", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("SI-SNR (dB)")
    plt.title("Training and Validation SI-SNR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

    for batch_idx, batch in enumerate(pbar):
        mix = batch["mix"].to(device)
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        batch_size = mix.size(0)

        mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
        ref_lengths = mix_lengths.clone()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, stats, weight = model(
                speech_mix=mix,
                speech_mix_lengths=mix_lengths,
                speech_ref1=s1,
                speech_ref1_lengths=ref_lengths,
                speech_ref2=s2,
                speech_ref2_lengths=ref_lengths,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN/Inf loss detected at batch {batch_idx}, skipping...")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        si_snr_db = -loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "SI-SNR": f"{si_snr_db:.2f} dB"})

    return total_loss / num_batches


def validate(model, val_loader, device, epoch):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            mix = batch["mix"].to(device)
            s1 = batch["s1"].to(device)
            s2 = batch["s2"].to(device)
            batch_size = mix.size(0)

            mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
            ref_lengths = mix_lengths.clone()

            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, stats, weight = model(
                    speech_mix=mix,
                    speech_mix_lengths=mix_lengths,
                    speech_ref1=s1,
                    speech_ref1_lengths=ref_lengths,
                    speech_ref2=s2,
                    speech_ref2_lengths=ref_lengths,
                )

            total_loss += loss.item()
            val_si_snr_db = -loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_SI-SNR": f"{val_si_snr_db:.2f} dB"})

    return total_loss / num_batches


def main(resume_from=None, num_epochs=None, dataset_dir=None, checkpoint_dir=None):
    global DATASET_DIR, CHECKPOINT_DIR
    if dataset_dir:
        DATASET_DIR = Path(dataset_dir).expanduser().resolve()
    if checkpoint_dir:
        CHECKPOINT_DIR = Path(checkpoint_dir).expanduser().resolve()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DATASET_DIR    = {DATASET_DIR}")
    print(f"CHECKPOINT_DIR = {CHECKPOINT_DIR}")

    random.seed(TRAIN_CONFIG["seed"])
    np.random.seed(TRAIN_CONFIG["seed"])
    torch.manual_seed(TRAIN_CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nBuilding utterance-level train/dev/test split...")
    train_utts, _, _ = build_utterance_split(
        RAW_DIR,
        seed=TRAIN_CONFIG["seed"],
        train_ratio=0.8,
        dev_ratio=0.1,
    )

    print("\nLoading datasets...")
    train_dataset = DynamicMixDataset(
        utterances_by_speaker=train_utts,
        num_speakers=2,
        target_duration=5.0,
        target_sr=16000,
        snr_range=(-5.0, 5.0),
        epoch_size=int(os.environ.get("TSS_TRAIN_EPOCH_SIZE", 28800)),
        gender_balance=True,
        augment=True,
    )
    dev_dataset = IndonesianMixDataset(
        split="dev",
        dataset_dir=DATASET_DIR,
        num_speakers=2,
        augment=False,
        target_duration=5.0,
    )
    test_dataset = IndonesianMixDataset(
        split="test",
        dataset_dir=DATASET_DIR,
        num_speakers=2,
        augment=False,
        target_duration=5.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches:   {len(dev_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    model = build_model(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-06, verbose=True
    )

    best_val_loss = float("inf")
    start_epoch = 1
    target_num_epochs = num_epochs if num_epochs is not None else TRAIN_CONFIG["num_epochs"]

    if resume_from is not None:
        resume_path = CHECKPOINT_DIR / resume_from
        print(f"\nLoading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}.")

    train_losses, val_losses = load_training_history(CHECKPOINT_DIR)
    if not train_losses and resume_from is not None:
        ckpt_ref = ckpt if resume_from is not None else {}
        train_losses = ckpt_ref.get("train_losses", [])
        val_losses = ckpt_ref.get("val_losses", [])
    if start_epoch > 1 and len(train_losses) >= start_epoch - 1:
        train_losses = train_losses[: start_epoch - 1]
        val_losses = val_losses[: start_epoch - 1]

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, target_num_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            train_losses.append(train_loss)

            val_loss = validate(model, dev_loader, device, epoch)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, Val SI-SNR = {-val_loss:.2f} dB"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "config": MODEL_CONFIG,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    CHECKPOINT_DIR / "best_model.pth",
                )
                print(f"  Best model saved (SI-SNR: {-val_loss:.2f} dB)")

            if epoch % 10 == 0:
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "config": MODEL_CONFIG,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    checkpoint_path,
                )
                print(f"  Checkpoint saved: epoch_{epoch}.pth")

            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            save_training_history(train_losses, val_losses, CHECKPOINT_DIR)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if train_losses:
            interrupted_path = CHECKPOINT_DIR / "checkpoint_interrupted.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_loss": train_losses[-1],
                    "val_loss": val_losses[-1] if val_losses else float("inf"),
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                },
                interrupted_path,
            )
            print(f"  Interrupted checkpoint saved: checkpoint_interrupted.pth (epoch {epoch})")

    finally:
        if train_losses:
            print("\n" + "=" * 60)
            print(f"Best validation SI-SNR: {-best_val_loss:.2f} dB")
            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            print("Training curves saved")
            with open(CHECKPOINT_DIR / "config.json", "w") as f:
                json.dump(
                    {
                        "model_config": MODEL_CONFIG,
                        "train_config": TRAIN_CONFIG,
                        "best_val_loss": best_val_loss,
                        "best_si_snr": -best_val_loss,
                    },
                    f,
                    indent=2,
                )
            print("Config saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SkiM Multi-Scale 2-Speaker model")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint filename to resume from (e.g. checkpoint_epoch_30.pth)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Total number of epochs to train (overrides config)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Override synthetic dataset dir (else $TSS_SYNTHETIC_DIR or project default)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint output dir (else $TSS_CHECKPOINT_DIR or project default)",
    )
    args = parser.parse_args()
    main(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
