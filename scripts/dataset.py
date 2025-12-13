#!/usr/bin/env python3
"""PyTorch Dataset for AutoHDR."""

import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PairedAugment:
    """Paired spatial augmentation for aligned input/target image pairs.

    Only uses spatial ops (crop/flip/rotate90) so the pixel-to-pixel mapping
    between input and target remains valid.
    """

    def __init__(self, crop_size: int | None = None, hflip_p: float = 0.5, vflip_p: float = 0.0, rotate90: bool = False):
        self.crop_size = crop_size
        self.hflip_p = float(hflip_p)
        self.vflip_p = float(vflip_p)
        self.rotate90 = bool(rotate90)

    def __call__(self, inp: Image.Image, tar: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.crop_size is not None:
            i, j, h, w = T.RandomCrop.get_params(inp, output_size=(self.crop_size, self.crop_size))
            inp = TF.crop(inp, i, j, h, w)
            tar = TF.crop(tar, i, j, h, w)

        if random.random() < self.hflip_p:
            inp = TF.hflip(inp)
            tar = TF.hflip(tar)

        if self.vflip_p > 0 and random.random() < self.vflip_p:
            inp = TF.vflip(inp)
            tar = TF.vflip(tar)

        if self.rotate90:
            k = random.randint(0, 3)
            if k:
                angle = 90 * k
                inp = TF.rotate(inp, angle, expand=False)
                tar = TF.rotate(tar, angle, expand=False)

        return inp, tar


class AutoHDRDataset(Dataset):
    """Dataset for paired real estate images."""
    
    def __init__(self, data_dir, transform=None, paired_transform=None):
        self.data_dir = Path(data_dir)
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
        
        self.images = sorted([
            f.stem for f in self.input_dir.iterdir()
            if f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        
        self.transform = transform or T.ToTensor()
        self.paired_transform = paired_transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        name = self.images[idx]
        
        inp = Image.open(self._find(self.input_dir, name)).convert("RGB")
        tar = Image.open(self._find(self.output_dir, name)).convert("RGB")
        
        # Ensure same size
        if inp.size != tar.size:
            tar = tar.resize(inp.size, Image.LANCZOS)

        if self.paired_transform is not None:
            inp, tar = self.paired_transform(inp, tar)
        
        return {
            "input": self.transform(inp),
            "target": self.transform(tar),
            "name": name
        }
    
    def _find(self, directory, stem):
        """Find file by stem with any extension."""
        for ext in [".png", ".jpg", ".jpeg"]:
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"{stem} not found in {directory}")


if __name__ == "__main__":
    # Test dataset
    import sys
    base = Path(__file__).parent.parent
    
    train_dir = base / "data/processed/train"
    if train_dir.exists():
        ds = AutoHDRDataset(train_dir)
        print(f"Dataset size: {len(ds)}")
        sample = ds[0]
        print(f"Input shape: {sample['input'].shape}")
        print(f"Target shape: {sample['target'].shape}")
        print(f"Name: {sample['name']}")
    else:
        print("Run preprocess.py first!")