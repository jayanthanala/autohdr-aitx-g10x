#!/usr/bin/env python3
"""
Batch inference with NAFNet.
Processes images and saves enhanced outputs.
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from model import NAFNet


def load_model(checkpoint_path, device):
    """Load trained NAFNet model."""
    model = NAFNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def process_image(model, img_path, device):
    """Process single image."""
    # Load
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size
    
    # Transform
    transform = T.ToTensor()
    inp = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = model(inp)
    
    # Post-process
    out = out.squeeze(0).cpu().clamp(0, 1)
    out_img = T.ToPILImage()(out)
    
    # Resize back to original
    out_img = out_img.resize(orig_size, Image.LANCZOS)
    
    return out_img


def main():
    parser = argparse.ArgumentParser(description="AutoHDR Inference")
    parser.add_argument("-i", "--input", required=True, 
                        help="Input image or directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")
    parser.add_argument("-c", "--checkpoint", 
                        default="models/checkpoints/best_model.pth",
                        help="Model checkpoint path")
    args = parser.parse_args()
    
    print("=" * 50)
    print("AutoHDR Inference - NAFNet")
    print("=" * 50)
    
    base = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    ckpt_path = base / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        print("Run training first!")
        return
    
    print(f"\nLoading model from {ckpt_path}...")
    model = load_model(ckpt_path, device)
    print("Model loaded!")
    
    # Get input images
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        images = [input_path]
    else:
        images = sorted(
            list(input_path.glob("*.jpg")) +
            list(input_path.glob("*.jpeg")) +
            list(input_path.glob("*.png")) +
            list(input_path.glob("*.JPG")) +
            list(input_path.glob("*.PNG"))
        )
    
    print(f"\nProcessing {len(images)} images...")
    
    times = []
    for img_path in tqdm(images, desc="Enhancing"):
        # Measure time
        start = time.time()
        
        # Process
        out_img = process_image(model, img_path, device)
        
        # Sync GPU for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
        
        # Save
        out_path = output_dir / f"{img_path.stem}_enhanced.png"
        out_img.save(out_path)
    
    # Stats
    avg_time = np.mean(times) * 1000
    print(f"\n" + "=" * 50)
    print("Inference complete!")
    print(f"Images processed: {len(images)}")
    print(f"Average time: {avg_time:.1f} ms/image")
    print(f"Output directory: {output_dir}")
    print("=" * 50)
    print(f"\nNext: python scripts/evaluate.py -p {output_dir} -g GROUND_TRUTH_DIR")


if __name__ == "__main__":
    main()