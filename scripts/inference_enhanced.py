#!/usr/bin/env python3
"""
Enhanced Inference with:
1. Sky Replacement - Replaces gray skies with blue
2. Multi-Resolution Ensemble - Process at multiple scales for better quality
3. Original aspect ratio preservation
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
from sky_replace import replace_sky, detect_sky_region


# Feature toggles
ENABLE_SKY_REPLACEMENT = True
ENABLE_MULTI_RESOLUTION = True
MULTI_RES_SCALES = [768, 1024, 1280]  # Process at multiple scales


def load_model(checkpoint_path, device):
    """Load trained NAFNet model."""
    model = NAFNet().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model


def process_single_scale(model, img_tensor, device):
    """Process image at single scale."""
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            output = model(img_tensor)
    return output


def process_multi_resolution(model, img, device, scales=MULTI_RES_SCALES):
    """
    Multi-Resolution Ensemble Processing.
    
    Process at multiple scales and blend for better quality:
    - Smaller scale: Better global adjustments
    - Larger scale: Better fine details
    
    Args:
        model: NAFNet model
        img: PIL Image (original)
        device: torch device
        scales: List of processing sizes
    
    Returns:
        PIL Image (enhanced)
    """
    transform = T.ToTensor()
    results = []
    weights = []
    
    original_size = img.size  # (W, H)
    
    for scale in scales:
        # Resize to square for model
        resized = img.resize((scale, scale), Image.LANCZOS)
        inp = transform(resized).unsqueeze(0).to(device)
        
        # Process
        out = process_single_scale(model, inp, device)
        out = out.squeeze(0).cpu().clamp(0, 1)
        out_img = T.ToPILImage()(out)
        
        # Resize to original dimensions for blending
        out_img = out_img.resize(original_size, Image.LANCZOS)
        results.append(np.array(out_img).astype(np.float32))
        
        # Higher resolution gets more weight
        weights.append(scale / max(scales))
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Weighted blend
    blended = np.zeros_like(results[0])
    for result, weight in zip(results, weights):
        blended += result * weight
    
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended)


def process_standard(model, img, device, process_size=1024):
    """Standard single-scale processing."""
    transform = T.ToTensor()
    original_size = img.size
    
    # Resize to model input size
    resized = img.resize((process_size, process_size), Image.LANCZOS)
    inp = transform(resized).unsqueeze(0).to(device)
    
    # Process
    out = process_single_scale(model, inp, device)
    out = out.squeeze(0).cpu().clamp(0, 1)
    out_img = T.ToPILImage()(out)
    
    # Resize back to original
    out_img = out_img.resize(original_size, Image.LANCZOS)
    
    return out_img


def process_image(model, img_path, device, use_multi_res=True, use_sky_replace=True):
    """
    Complete image processing pipeline.
    
    Args:
        model: NAFNet model
        img_path: Path to input image
        device: torch device
        use_multi_res: Enable multi-resolution ensemble
        use_sky_replace: Enable sky replacement
    
    Returns:
        PIL Image (enhanced)
    """
    # Load image
    img = Image.open(img_path).convert("RGB")
    original_size = img.size
    
    # Step 1: NAFNet Enhancement
    if use_multi_res and ENABLE_MULTI_RESOLUTION:
        enhanced = process_multi_resolution(model, img, device)
    else:
        enhanced = process_standard(model, img, device)
    
    # Step 2: Sky Replacement (if enabled)
    if use_sky_replace and ENABLE_SKY_REPLACEMENT:
        # Check if sky needs replacement
        sky_mask = detect_sky_region(img)
        sky_coverage = sky_mask.mean()
        
        if sky_coverage > 0.02:  # More than 2% sky detected
            enhanced = replace_sky(enhanced)
    
    return enhanced


def process_batch(model, input_dir, output_dir, device, use_multi_res=True, use_sky_replace=True):
    """Process batch of images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    images = []
    for ext in extensions:
        images.extend(input_path.glob(f"*{ext}"))
    images = sorted(images)
    
    print(f"\nProcessing {len(images)} images...")
    print(f"Multi-Resolution: {'ON' if use_multi_res else 'OFF'}")
    print(f"Sky Replacement: {'ON' if use_sky_replace else 'OFF'}")
    
    times = []
    
    for img_path in tqdm(images, desc="Enhancing"):
        start = time.time()
        
        # Process
        out_img = process_image(
            model, img_path, device,
            use_multi_res=use_multi_res,
            use_sky_replace=use_sky_replace
        )
        
        # Sync GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append(time.time() - start)
        
        # Save
        out_path = output_path / f"{img_path.stem}_enhanced.png"
        out_img.save(out_path)
    
    return times


def main():
    parser = argparse.ArgumentParser(description="AutoHDR Enhanced Inference")
    parser.add_argument("-i", "--input", required=True,
                        help="Input image or directory")
    parser.add_argument("-o", "--output", required=True,
                        help="Output directory")
    parser.add_argument("-c", "--checkpoint",
                        default="models/checkpoints/best_model.pth",
                        help="Model checkpoint path")
    parser.add_argument("--no-multires", action="store_true",
                        help="Disable multi-resolution ensemble")
    parser.add_argument("--no-sky", action="store_true",
                        help="Disable sky replacement")
    args = parser.parse_args()
    
    print("=" * 50)
    print("AutoHDR Enhanced Inference")
    print("=" * 50)
    
    # Feature status
    use_multi_res = not args.no_multires
    use_sky_replace = not args.no_sky
    
    print(f"\nFeatures:")
    print(f"  Multi-Resolution Ensemble: {'ON' if use_multi_res else 'OFF'}")
    print(f"  Sky Replacement: {'ON' if use_sky_replace else 'OFF'}")
    
    base = Path(__file__).parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load model
    ckpt_path = base / args.checkpoint
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return
    
    print(f"\nLoading model from {ckpt_path}...")
    model = load_model(ckpt_path, device)
    print("Model loaded!")
    
    # Process
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        print(f"\nProcessing single image: {input_path}")
        start = time.time()
        
        out_img = process_image(
            model, input_path, device,
            use_multi_res=use_multi_res,
            use_sky_replace=use_sky_replace
        )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        proc_time = (time.time() - start) * 1000
        
        out_path = output_dir / f"{input_path.stem}_enhanced.png"
        out_img.save(out_path)
        
        print(f"Saved: {out_path}")
        print(f"Time: {proc_time:.1f} ms")
    else:
        # Batch processing
        times = process_batch(
            model, input_path, output_dir, device,
            use_multi_res=use_multi_res,
            use_sky_replace=use_sky_replace
        )
        
        avg_time = np.mean(times) * 1000
        
        print(f"\n" + "=" * 50)
        print("Inference complete!")
        print(f"Images processed: {len(times)}")
        print(f"Average time: {avg_time:.1f} ms/image")
        print(f"Output directory: {output_dir}")
        print("=" * 50)
        print(f"\nNext: python scripts/evaluate.py -p {output_dir} -g GROUND_TRUTH_DIR")


if __name__ == "__main__":
    main()