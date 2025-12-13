#!/usr/bin/env python3
"""
Preprocess AutoHDR dataset.
- Resolution: 1024x1024 (fixed size for batching)
- Format: XXX_src.jpg (input) / XXX_tar.jpg (target)
- Split: 90/10 train/val
- Data leakage prevention with fixed seed
"""

import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

RANDOM_SEED = 42
TRAIN_SPLIT = 0.9
TARGET_SIZE = 1024


def get_image_pairs(raw_dir):
    """Find all src/tar pairs."""
    raw_path = Path(raw_dir)
    images_dir = raw_path / "images" if (raw_path / "images").exists() else raw_path
    
    print(f"Scanning: {images_dir}")
    
    pairs = []
    for src_file in sorted(images_dir.glob("*_src.*")):
        base = src_file.stem.replace("_src", "")
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            tar_file = images_dir / f"{base}_tar{ext}"
            if tar_file.exists():
                pairs.append((src_file, tar_file))
                break
    
    return pairs


def create_split(pairs):
    """Split with NO data leakage."""
    random.seed(RANDOM_SEED)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    
    idx = int(len(shuffled) * TRAIN_SPLIT)
    train, val = shuffled[:idx], shuffled[idx:]
    
    # Verify no leakage
    train_ids = {p[0].stem.replace("_src", "") for p in train}
    val_ids = {p[0].stem.replace("_src", "") for p in val}
    overlap = train_ids & val_ids
    
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE DETECTED: {overlap}")
    
    print(f"Split: {len(train)} train, {len(val)} val (seed={RANDOM_SEED})")
    print(f"Data leakage check: PASSED (0 overlap)")
    return train, val


def resize_image(path, size):
    """Resize to fixed size (1024x1024) for consistent batching."""
    img = Image.open(path).convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


def process_pairs(pairs, out_dir):
    """Process and save pairs."""
    inp_dir = Path(out_dir) / "input"
    tar_dir = Path(out_dir) / "output"
    inp_dir.mkdir(parents=True, exist_ok=True)
    tar_dir.mkdir(parents=True, exist_ok=True)
    
    for src, tar in tqdm(pairs, desc=f"Processing {Path(out_dir).name}"):
        base_id = src.stem.replace("_src", "")
        resize_image(src, TARGET_SIZE).save(inp_dir / f"{base_id}.png")
        resize_image(tar, TARGET_SIZE).save(tar_dir / f"{base_id}.png")


def main():
    base = Path(__file__).parent.parent
    
    print("=" * 50)
    print(f"AutoHDR Preprocessing")
    print(f"Resolution: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Split: {int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}")
    print("=" * 50)
    
    pairs = get_image_pairs(base / "data/raw")
    print(f"\nFound {len(pairs)} pairs")
    
    if not pairs:
        print("\nERROR: No pairs found!")
        print("Expected: data/raw/images/XXX_src.jpg + XXX_tar.jpg")
        return
    
    train, val = create_split(pairs)
    process_pairs(train, base / "data/processed/train")
    process_pairs(val, base / "data/processed/val")
    
    # Save manifest
    with open(base / "data/split_manifest.txt", "w") as f:
        f.write(f"# AutoHDR Split Manifest\n")
        f.write(f"# Seed: {RANDOM_SEED}\n")
        f.write(f"# Resolution: {TARGET_SIZE}x{TARGET_SIZE}\n")
        f.write(f"# Train: {len(train)}, Val: {len(val)}\n\n")
        for src, _ in train:
            f.write(f"train:{src.stem.replace('_src', '')}\n")
        for src, _ in val:
            f.write(f"val:{src.stem.replace('_src', '')}\n")
    
    print(f"\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"Train: {base / 'data/processed/train'}")
    print(f"Val: {base / 'data/processed/val'}")
    print("=" * 50)
    print(f"\nNext: python scripts/train.py")


if __name__ == "__main__":
    main()