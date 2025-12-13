#!/usr/bin/env python3
"""
NAFNet Fine-tuning for Better Color/Brightness (Higher PSNR).

Changes from original training:
- Loads existing checkpoint
- Disables cropping (full 1024x1024)
- Adds color loss for better color matching
- Lower learning rate
- Fewer epochs
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AutoHDRDataset
from model import NAFNet, count_parameters


# Fine-tuning Configuration
CONFIG = {
    "batch_size": 1,        # Full 1024x1024 needs batch=1
    "lr": 5e-5,             # Lower LR for fine-tuning
    "epochs": 20,           # Quick fine-tune
    "save_every": 5,
    "workers": 18,
}


class L1SSIMColorLoss(nn.Module):
    """
    L1 + SSIM + Color loss for better color/brightness matching.
    
    - L1: Pixel accuracy
    - SSIM: Structure preservation  
    - Color: Global color mean/std matching (BOOSTS PSNR!)
    """
    
    def __init__(self, l1_weight=1.5, ssim_weight=0.1, color_weight=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
    
    def forward(self, pred, target):
        # L1 loss
        l1_loss = self.l1(pred, target)
        
        # SSIM loss
        mu_p = pred.mean(dim=[2, 3], keepdim=True)
        mu_t = target.mean(dim=[2, 3], keepdim=True)
        sigma_p = ((pred - mu_p) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_t = ((target - mu_t) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=[2, 3], keepdim=True)
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
        ssim_loss = 1 - ssim.mean()
        
        # Color loss - match global color means (per channel)
        pred_mean = pred.mean(dim=[2, 3])  # [B, 3]
        target_mean = target.mean(dim=[2, 3])
        color_mean_loss = torch.abs(pred_mean - target_mean).mean()
        
        # Color std loss - match color variance
        pred_std = pred.std(dim=[2, 3])
        target_std = target.std(dim=[2, 3])
        color_std_loss = torch.abs(pred_std - target_std).mean()
        
        total_color_loss = color_mean_loss + color_std_loss
        
        return (self.l1_weight * l1_loss + 
                self.ssim_weight * ssim_loss + 
                self.color_weight * total_color_loss)


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        return device, True
    return torch.device("cpu"), False


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Train")
    for batch in pbar:
        inputs = batch["input"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, use_amp):
    """Validate model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            inputs = batch["input"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            
            if use_amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    print("=" * 60)
    print("  AutoHDR Fine-tuning - Boost PSNR")
    print("=" * 60)
    
    base = Path(__file__).parent.parent
    device, use_amp = get_device()
    
    print(f"\nDevice: {device}")
    print(f"AMP: {'ON' if use_amp else 'OFF'}")
    
    # Load data - NO AUGMENTATION for fine-tuning
    print(f"\nLoading data (full 1024x1024, no crop)...")
    train_data = AutoHDRDataset(base / "data/processed/train")  # No augmentation!
    val_data = AutoHDRDataset(base / "data/processed/val")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    loader_kwargs = {
        "batch_size": CONFIG["batch_size"],
        "num_workers": CONFIG["workers"] if device.type == "cuda" else 0,
    }
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
        loader_kwargs["persistent_workers"] = True
    
    train_loader = DataLoader(train_data, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **loader_kwargs)
    
    # Model
    print(f"\nInitializing NAFNet...")
    model = NAFNet().to(device)
    
    # LOAD EXISTING CHECKPOINT
    ckpt_path = base / "models/checkpoints/best_model.pth"
    if ckpt_path.exists():
        print(f"Loading checkpoint for fine-tuning: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        prev_loss = checkpoint.get("val_loss", "unknown")
        print(f"  ✓ Previous val_loss: {prev_loss}")
    else:
        print("  ⚠ No checkpoint found, training from scratch")
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Use Color Loss!
    print("\nUsing L1 + SSIM + Color loss for better color matching")
    criterion = L1SSIMColorLoss(l1_weight=1.5, ssim_weight=0.1, color_weight=0.5)
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )
    
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # Checkpoints
    best_loss = float("inf")
    ckpt_dir = base / "models/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup original
    if ckpt_path.exists():
        backup_path = ckpt_dir / "best_model_before_finetune.pth"
        if not backup_path.exists():
            import shutil
            shutil.copy(ckpt_path, backup_path)
            print(f"  ✓ Backed up to: {backup_path}")
    
    print(f"\nStarting fine-tuning...")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG,
                "fine_tuned": True,
            }, ckpt_dir / "best_model.pth")
            print(">>> Saved best model!")
        
        if (epoch + 1) % CONFIG["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, ckpt_dir / f"finetune_epoch_{epoch + 1}.pth")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Fine-tuning complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best val loss: {best_loss:.4f}")
    print("=" * 60)
    print("\nNow re-run inference and evaluation:")
    print("  python scripts/inference.py -i data/processed/val/input -o outputs/")
    print("  python scripts/evaluate.py -p outputs/ -g data/processed/val/output")


if __name__ == "__main__":
    main()