#!/usr/bin/env python3
"""
NAFNet Training with Mixed Precision (AMP).

Features:
- Automatic Mixed Precision (FP16) for faster training
- L1 + SSIM loss for perceptual quality
- Cosine annealing learning rate
- Best model checkpointing
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


# Configuration
CONFIG = {
    "batch_size": 2,        # Fits 1024x1024 in VRAM
    "lr": 1e-4,             # Learning rate
    "epochs": 50,           # Training epochs
    "save_every": 10,       # Save checkpoint every N epochs
    "workers": 4,           # DataLoader workers
}


class L1SSIMLoss(nn.Module):
    """
    Combined L1 + SSIM loss.
    L1 for pixel accuracy, SSIM for perceptual quality.
    """
    
    def __init__(self, l1_weight=1.0, ssim_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
    
    def forward(self, pred, target):
        # L1 loss
        l1_loss = self.l1(pred, target)
        
        # Simplified SSIM loss
        mu_p = pred.mean(dim=[2, 3], keepdim=True)
        mu_t = target.mean(dim=[2, 3], keepdim=True)
        sigma_p = ((pred - mu_p) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_t = ((target - mu_t) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_pt = ((pred - mu_p) * (target - mu_t)).mean(dim=[2, 3], keepdim=True)
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p + sigma_t + C2))
        ssim_loss = 1 - ssim.mean()
        
        return self.l1_weight * l1_loss + self.ssim_weight * ssim_loss


def get_device():
    """Get the best available device with detailed diagnostics."""
    print("\nChecking available devices...")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  ✓ CUDA available")
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ GPU Memory: {mem:.1f} GB")
        return device, True
    else:
        print(f"  ✗ CUDA not available!")
        print(f"  → Checking why...")
        print(f"    torch.version.cuda = {torch.version.cuda}")
        print(f"    torch.backends.cudnn.enabled = {torch.backends.cudnn.enabled}")
        
        # Try to get more info
        try:
            import subprocess
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  → nvidia-smi works, but PyTorch can't see GPU")
                print(f"  → Try: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            else:
                print(f"  → nvidia-smi failed: {result.stderr}")
        except Exception as e:
            print(f"  → Could not run nvidia-smi: {e}")
        
        print(f"\n  ⚠ Falling back to CPU (will be slow!)")
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
        
        # Mixed precision forward pass (updated API)
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
    print("  AutoHDR Training - NAFNet")
    print("=" * 60)
    
    base = Path(__file__).parent.parent
    
    # Get device with diagnostics
    device, use_amp = get_device()
    
    print(f"\nDevice: {device}")
    print(f"Mixed Precision (AMP): {'ON' if use_amp else 'OFF'}")
    
    # Load data
    print(f"\nLoading data...")
    train_data = AutoHDRDataset(base / "data/processed/train")
    val_data = AutoHDRDataset(base / "data/processed/val")
    print(f"Train: {len(train_data)} | Val: {len(val_data)}")
    
    # DataLoader settings depend on device
    loader_kwargs = {
        "batch_size": CONFIG["batch_size"],
        "num_workers": CONFIG["workers"] if device.type == "cuda" else 0,
    }
    
    if device.type == "cuda":
        loader_kwargs["pin_memory"] = True
        loader_kwargs["persistent_workers"] = True
    
    train_loader = DataLoader(
        train_data,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_data,
        shuffle=False,
        **loader_kwargs
    )
    
    # Model
    print(f"\nInitializing NAFNet...")
    model = NAFNet().to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            test_input = torch.randn(1, 3, 64, 64).to(device)
            test_output = model(test_input)
        print(f"  ✓ Model works: {test_input.shape} -> {test_output.shape}")
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return
    
    # Training setup
    criterion = L1SSIMLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )
    
    # GradScaler for AMP (updated API)
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None
    
    # Checkpoints
    best_loss = float("inf")
    ckpt_dir = base / "models/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Training log
    log_dir = base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training...")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print("-" * 60)
    
    start_time = time.time()
    history = []
    
    for epoch in range(CONFIG["epochs"]):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device, use_amp)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {lr:.2e} | Time: {epoch_time:.1f}s")
        
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "time": epoch_time
        })
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": CONFIG,
            }, ckpt_dir / "best_model.pth")
            print(">>> Saved best model!")
        
        # Periodic checkpoint
        if (epoch + 1) % CONFIG["save_every"] == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            }, ckpt_dir / f"ckpt_epoch_{epoch + 1}.pth")
    
    total_time = time.time() - start_time
    
    # Save training log
    with open(log_dir / "training_log.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("AutoHDR Training Log - NAFNet\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total time: {total_time / 60:.1f} minutes\n")
        f.write(f"Best val loss: {best_loss:.4f}\n")
        f.write(f"Device: {device}\n")
        f.write(f"AMP: {use_amp}\n\n")
        f.write("Epoch History:\n")
        f.write("-" * 60 + "\n")
        for h in history:
            f.write(f"Epoch {h['epoch']:3d}: train={h['train_loss']:.4f}, "
                    f"val={h['val_loss']:.4f}, lr={h['lr']:.2e}, "
                    f"time={h['time']:.1f}s\n")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Best val loss: {best_loss:.4f}")
    print(f"Model saved: {ckpt_dir / 'best_model.pth'}")
    print("=" * 60)
    print(f"\nNext: python scripts/inference.py -i INPUT -o OUTPUT")


if __name__ == "__main__":
    main()