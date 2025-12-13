#!/usr/bin/env python3
"""
Sky Replacement Module for AutoHDR.

Detects gray/overcast skies and replaces with blue sky gradient.
This addresses the model's weakness in sky enhancement.
"""

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Note: scipy not installed, using PIL blur instead")


def detect_sky_region(image, threshold_brightness=180, threshold_saturation=40):
    """
    Detect sky region using color and position heuristics.
    
    Sky characteristics:
    - Usually in top portion of image
    - High brightness (gray/white overcast OR blue)
    - Low saturation (gray) or specific blue hue
    
    Args:
        image: PIL Image
        threshold_brightness: Min brightness to consider as sky (0-255)
        threshold_saturation: Max saturation to consider as gray sky (0-255)
    
    Returns:
        numpy array mask (0-1 float), same size as image
    """
    img_array = np.array(image).astype(np.float32)
    height, width = img_array.shape[:2]
    
    # Convert to HSV-like metrics
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    
    # Brightness (simple average)
    brightness = (r + g + b) / 3
    
    # Saturation approximation
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = max_rgb - min_rgb
    
    # Detect potential sky pixels
    # Gray sky: bright + low saturation
    gray_sky_mask = (brightness > threshold_brightness) & (saturation < threshold_saturation)
    
    # Also detect already-blue sky (high blue, lower red)
    blue_sky_mask = (b > r + 20) & (b > g) & (brightness > 120)
    
    # Combine sky detection
    sky_mask = gray_sky_mask | blue_sky_mask
    
    # Apply position weight - sky more likely at top
    position_weight = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
        # Weight decreases as we go down
        weight = max(0, 1 - (y / (height * 0.6)))  # Top 60% gets weight
        position_weight[y, :] = weight
    
    # Combine detection with position
    sky_mask = sky_mask.astype(np.float32) * position_weight
    
    # Clean up mask - remove small isolated regions
    sky_mask = _clean_mask(sky_mask)
    
    # Smooth edges for natural blending
    sky_mask = _smooth_mask(sky_mask, sigma=15)
    
    return sky_mask


def _clean_mask(mask, min_region_ratio=0.01):
    """Remove small isolated regions from mask."""
    # Simple threshold to binary
    binary = (mask > 0.5).astype(np.uint8)
    
    # If sky region too small, likely false detection
    sky_ratio = binary.sum() / binary.size
    if sky_ratio < min_region_ratio:
        return np.zeros_like(mask)
    
    return mask


def _smooth_mask(mask, sigma=15):
    """Smooth mask edges for natural blending."""
    if HAS_SCIPY:
        return gaussian_filter(mask, sigma=sigma)
    else:
        # Use PIL blur as fallback
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.array(mask_img).astype(np.float32) / 255


def generate_blue_sky(size, style="gradient"):
    """
    Generate a realistic blue sky image.
    
    Args:
        size: (width, height) tuple
        style: "gradient" | "clouds" | "clear"
    
    Returns:
        PIL Image of blue sky
    """
    width, height = size
    sky = np.zeros((height, width, 3), dtype=np.uint8)
    
    if style == "gradient":
        # Natural gradient: deeper blue at top, lighter near horizon
        for y in range(height):
            ratio = y / height
            
            # Sky blue to light blue/white gradient
            r = int(100 + ratio * 140)   # 100 -> 240
            g = int(160 + ratio * 90)    # 160 -> 250
            b = int(220 + ratio * 35)    # 220 -> 255
            
            # Clamp values
            r = min(255, max(0, r))
            g = min(255, max(0, g))
            b = min(255, max(0, b))
            
            sky[y, :] = [r, g, b]
    
    elif style == "clear":
        # Uniform clear blue
        sky[:, :] = [135, 206, 235]  # Sky blue
    
    elif style == "clouds":
        # Gradient with slight variation (fake clouds)
        for y in range(height):
            ratio = y / height
            base_r = int(100 + ratio * 155)
            base_g = int(160 + ratio * 95)
            base_b = int(220 + ratio * 35)
            
            for x in range(width):
                # Add slight variation
                noise = np.sin(x * 0.01 + y * 0.02) * 10
                r = min(255, max(0, int(base_r + noise)))
                g = min(255, max(0, int(base_g + noise)))
                b = min(255, max(0, int(base_b + noise * 0.5)))
                sky[y, x] = [r, g, b]
    
    return Image.fromarray(sky)


def replace_sky(image, sky_style="gradient", blend_strength=0.95):
    """
    Replace gray/overcast sky with blue sky.
    
    Args:
        image: PIL Image (enhanced by NAFNet)
        sky_style: "gradient" | "clouds" | "clear"
        blend_strength: How much to replace (0-1)
    
    Returns:
        PIL Image with replaced sky
    """
    # Detect sky region
    sky_mask = detect_sky_region(image)
    
    # Check if significant sky found
    sky_coverage = sky_mask.mean()
    if sky_coverage < 0.01:  # Less than 1% sky
        # No sky detected, return original
        return image
    
    # Generate blue sky
    blue_sky = generate_blue_sky(image.size, style=sky_style)
    
    # Convert to arrays
    img_array = np.array(image).astype(np.float32)
    sky_array = np.array(blue_sky).astype(np.float32)
    
    # Apply blend strength
    effective_mask = sky_mask * blend_strength
    
    # Expand mask to 3 channels
    mask_3d = effective_mask[:, :, np.newaxis]
    
    # Blend
    result = img_array * (1 - mask_3d) + sky_array * mask_3d
    
    # Ensure valid range
    result = np.clip(result, 0, 255)
    
    return Image.fromarray(result.astype(np.uint8))


def enhance_with_sky_replacement(image, model_output, auto_detect=True):
    """
    Complete sky replacement pipeline.
    
    Args:
        image: Original input image (for reference)
        model_output: NAFNet enhanced image
        auto_detect: Whether to auto-detect if sky needs replacement
    
    Returns:
        Final enhanced image with sky replacement
    """
    if auto_detect:
        # Check if original has gray sky
        original_mask = detect_sky_region(image)
        
        # If no significant sky in original, skip
        if original_mask.mean() < 0.02:
            return model_output
        
        # Check if model output still has gray sky
        output_array = np.array(model_output)
        if output_array.size == 0:
            return model_output
            
        # Sample sky region colors from output
        sky_region = output_array[original_mask > 0.5]
        if len(sky_region) == 0:
            return model_output
            
        # Calculate average blue ratio
        avg_color = sky_region.mean(axis=0)
        blue_ratio = avg_color[2] / (avg_color.mean() + 1e-6)
        
        # If sky is already blue enough, skip replacement
        if blue_ratio > 1.15:  # Blue channel 15% higher than average
            return model_output
    
    # Apply sky replacement
    return replace_sky(model_output)


# Test function
if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("Sky Replacement Module Test")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    else:
        test_image = None
    
    if test_image:
        print(f"\nLoading: {test_image}")
        img = Image.open(test_image).convert("RGB")
        
        print(f"Size: {img.size}")
        
        # Detect sky
        mask = detect_sky_region(img)
        print(f"Sky coverage: {mask.mean() * 100:.1f}%")
        
        # Replace sky
        result = replace_sky(img)
        
        # Save result
        out_path = test_image.replace(".", "_sky_replaced.")
        result.save(out_path)
        print(f"Saved: {out_path}")
    else:
        print("\nUsage: python sky_replace.py <image_path>")
        print("\nGenerating sample blue sky...")
        sky = generate_blue_sky((512, 256), style="gradient")
        sky.save("sample_blue_sky.png")
        print("Saved: sample_blue_sky.png")