#!/usr/bin/env python3
"""
FastAPI server for AutoHDR inference.
Provides REST API for image enhancement.
"""

import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from PIL import Image
import torchvision.transforms as T

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse
except ImportError:
    print("Installing FastAPI...")
    import subprocess
    subprocess.run(["pip", "install", "fastapi", "uvicorn", "python-multipart", "-q"])
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse

from model import NAFNet, count_parameters


# Global model
app = FastAPI(
    title="AutoHDR API",
    description="Real estate photo enhancement powered by NAFNet",
    version="1.0.0"
)

MODEL = None
DEVICE = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global MODEL, DEVICE
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    MODEL = NAFNet().to(DEVICE)
    
    # Load checkpoint
    ckpt_path = Path(__file__).parent.parent / "models/checkpoints/best_model.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Warning: No checkpoint found at {ckpt_path}")
        print("Model initialized with random weights.")
    
    MODEL.eval()
    print(f"NAFNet ready! ({count_parameters(MODEL):,} params)")


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "AutoHDR API",
        "model": "NAFNet",
        "device": str(DEVICE),
        "endpoints": {
            "/health": "Health check",
            "/info": "Model info",
            "/enhance": "POST image for enhancement",
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available()
    }


@app.get("/info")
async def info():
    """Model information."""
    info = {
        "model": "NAFNet",
        "parameters": f"{count_parameters(MODEL):,}" if MODEL else "N/A",
        "device": str(DEVICE),
        "input_format": "RGB image (any size)",
        "output_format": "PNG image (same size as input)",
    }
    
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
    
    return info


@app.post("/enhance")
async def enhance(image: UploadFile = File(...)):
    """
    Enhance a real estate photo.
    
    - **image**: Image file (JPEG, PNG)
    - Returns: Enhanced image as PNG
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read image
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    orig_size = img.size
    
    # Transform
    transform = T.ToTensor()
    inp = transform(img).unsqueeze(0).to(DEVICE)
    
    # Inference
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            out = MODEL(inp)
    
    inference_time = (time.time() - start) * 1000
    
    # Post-process
    out = out.squeeze(0).cpu().clamp(0, 1)
    out_img = T.ToPILImage()(out)
    out_img = out_img.resize(orig_size, Image.LANCZOS)
    
    # Return as PNG
    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    
    print(f"Enhanced {image.filename} in {inference_time:.1f}ms")
    
    return StreamingResponse(
        buf, 
        media_type="image/png",
        headers={
            "X-Inference-Time-Ms": str(round(inference_time, 1)),
            "X-Original-Size": f"{orig_size[0]}x{orig_size[1]}"
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("AutoHDR API Server")
    print("=" * 50)
    print("\nStarting server on http://0.0.0.0:8000")
    print("\nEndpoints:")
    print("  GET  /        - API info")
    print("  GET  /health  - Health check")
    print("  GET  /info    - Model info")
    print("  POST /enhance - Enhance image")
    print("\nExample:")
    print("  curl -X POST -F 'image=@photo.jpg' http://localhost:8000/enhance -o enhanced.png")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)