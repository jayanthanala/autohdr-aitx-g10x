#!/usr/bin/env python3
"""
Evaluate model outputs using PSNR and SSIM metrics.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
except ImportError:
    print("Installing scikit-image...")
    import subprocess
    subprocess.run(["pip", "install", "scikit-image", "-q"])
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def load_image(path):
    """Load image as numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def compute_metrics(pred_img, gt_img):
    """Compute PSNR and SSIM."""
    # Ensure same size
    if pred_img.shape != gt_img.shape:
        gt_pil = Image.fromarray(gt_img)
        gt_pil = gt_pil.resize((pred_img.shape[1], pred_img.shape[0]), Image.LANCZOS)
        gt_img = np.array(gt_pil)
    
    psnr = peak_signal_noise_ratio(gt_img, pred_img, data_range=255)
    ssim = structural_similarity(gt_img, pred_img, channel_axis=2, data_range=255)
    
    return psnr, ssim


def main():
    parser = argparse.ArgumentParser(description="Evaluate AutoHDR outputs")
    parser.add_argument("-p", "--predictions", required=True,
                        help="Directory with predicted images")
    parser.add_argument("-g", "--ground-truth", required=True,
                        help="Directory with ground truth images")
    args = parser.parse_args()
    
    print("=" * 50)
    print("AutoHDR Evaluation")
    print("=" * 50)
    
    pred_dir = Path(args.predictions)
    gt_dir = Path(args.ground_truth)
    
    # Find prediction files
    pred_files = sorted(
        list(pred_dir.glob("*_enhanced.png")) +
        list(pred_dir.glob("*_enhanced.jpg"))
    )
    
    if not pred_files:
        # Try without _enhanced suffix
        pred_files = sorted(
            list(pred_dir.glob("*.png")) +
            list(pred_dir.glob("*.jpg"))
        )
    
    print(f"\nFound {len(pred_files)} predictions")
    print(f"Ground truth: {gt_dir}")
    
    psnr_scores = []
    ssim_scores = []
    results = []
    
    for pred_path in tqdm(pred_files, desc="Evaluating"):
        # Extract base name
        name = pred_path.stem.replace("_enhanced", "")
        
        # Find matching ground truth
        gt_path = None
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            candidate = gt_dir / f"{name}{ext}"
            if candidate.exists():
                gt_path = candidate
                break
        
        if not gt_path:
            print(f"  Warning: No ground truth for {name}")
            continue
        
        # Load images
        pred_img = load_image(pred_path)
        gt_img = load_image(gt_path)
        
        # Compute metrics
        psnr, ssim = compute_metrics(pred_img, gt_img)
        
        psnr_scores.append(psnr)
        ssim_scores.append(ssim)
        results.append({"name": name, "psnr": psnr, "ssim": ssim})
    
    # Summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nImages evaluated: {len(psnr_scores)}")
    print(f"\nPSNR:")
    print(f"  Mean:   {np.mean(psnr_scores):.2f} dB")
    print(f"  Std:    {np.std(psnr_scores):.2f} dB")
    print(f"  Min:    {np.min(psnr_scores):.2f} dB")
    print(f"  Max:    {np.max(psnr_scores):.2f} dB")
    print(f"\nSSIM:")
    print(f"  Mean:   {np.mean(ssim_scores):.4f}")
    print(f"  Std:    {np.std(ssim_scores):.4f}")
    print(f"  Min:    {np.min(ssim_scores):.4f}")
    print(f"  Max:    {np.max(ssim_scores):.4f}")
    print("=" * 50)
    
    # Save results
    base = Path(__file__).parent.parent
    results_path = base / "benchmarks" / "evaluation_results.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        f.write("AutoHDR Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Images: {len(psnr_scores)}\n")
        f.write(f"PSNR Mean: {np.mean(psnr_scores):.2f} dB\n")
        f.write(f"SSIM Mean: {np.mean(ssim_scores):.4f}\n\n")
        f.write("Per-image results:\n")
        for r in sorted(results, key=lambda x: x['psnr']):
            f.write(f"  {r['name']}: PSNR={r['psnr']:.2f}, SSIM={r['ssim']:.4f}\n")
    
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()