#!/usr/bin/env python3
"""
AutoHDR Server - Shows Enhanced Image Only.

Pipeline:
1. NAFNet: Base image enhancement (trained model)
2. Sky Replacement: Local sky detection + blue sky (optional)
3. Color Enhancement: PIL-based brightness/saturation boost
"""

import io
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms as T

try:
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
    from contextlib import asynccontextmanager
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "fastapi", "uvicorn", "python-multipart", "-q"])
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
    from contextlib import asynccontextmanager

from model import NAFNet, count_parameters

# Try to import sky replacement
try:
    from sky_replace import replace_sky
    SKY_REPLACE_AVAILABLE = True
except ImportError:
    SKY_REPLACE_AVAILABLE = False
    print("Note: sky_replace.py not found, sky replacement disabled")


# ============================================
# DEFAULT ENHANCEMENT SETTINGS
# ============================================

DEFAULTS = {
    "brightness": 1.0,
    "contrast": 1.0,
    "saturation": 1.0,
    "sharpness": 1.0,
    "sky_replacement": False,
}


# ============================================
# GLOBAL VARIABLES
# ============================================

MODEL = None
DEVICE = None


# ============================================
# LIFESPAN
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global MODEL, DEVICE
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    
    MODEL = NAFNet().to(DEVICE)
    
    ckpt_path = Path(__file__).parent.parent / "models/checkpoints/best_model.pth"
    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        MODEL.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ NAFNet loaded: {ckpt_path}")
    else:
        print(f"⚠ No checkpoint found")
    
    MODEL.eval()
    print(f"✓ NAFNet ready ({count_parameters(MODEL):,} params)")
    print(f"✓ Sky replacement: {'Available' if SKY_REPLACE_AVAILABLE else 'Disabled'}")
    
    yield
    print("Shutting down...")


# ============================================
# FASTAPI APP
# ============================================

app = FastAPI(
    title="AutoHDR Professional",
    description="AI Photo Enhancement",
    version="3.0.0",
    lifespan=lifespan
)


# ============================================
# PROCESSING FUNCTIONS
# ============================================

def process_nafnet(img):
    """Run NAFNet enhancement."""
    transform = T.ToTensor()
    original_size = img.size
    
    resized = img.resize((1024, 1024), Image.LANCZOS)
    inp = transform(resized).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
            out = MODEL(inp)
    
    out = out.squeeze(0).cpu().clamp(0, 1)
    out_img = T.ToPILImage()(out)
    
    return out_img.resize(original_size, Image.LANCZOS)


def apply_color_enhancement(img):
    """Apply color enhancements using PIL."""
    result = img
    
    if DEFAULTS["brightness"] != 1.0:
        result = ImageEnhance.Brightness(result).enhance(DEFAULTS["brightness"])
    
    if DEFAULTS["contrast"] != 1.0:
        result = ImageEnhance.Contrast(result).enhance(DEFAULTS["contrast"])
    
    if DEFAULTS["saturation"] != 1.0:
        result = ImageEnhance.Color(result).enhance(DEFAULTS["saturation"])
    
    if DEFAULTS["sharpness"] != 1.0:
        result = ImageEnhance.Sharpness(result).enhance(DEFAULTS["sharpness"])
    
    return result


def process_image(img):
    """Complete enhancement pipeline."""
    timings = {}
    
    # Stage 1: NAFNet
    start = time.time()
    result = process_nafnet(img)
    timings["nafnet"] = time.time() - start
    
    # Stage 2: Sky Replacement (if enabled)
    if DEFAULTS["sky_replacement"] and SKY_REPLACE_AVAILABLE:
        start = time.time()
        result = replace_sky(result)
        timings["sky_replace"] = time.time() - start
    
    # Stage 3: Color Enhancement
    start = time.time()
    result = apply_color_enhancement(result)
    timings["color"] = time.time() - start
    
    return result, timings


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
async def root():
    """API info."""
    return {
        "name": "AutoHDR Professional",
        "version": "3.0.0",
        "endpoints": {
            "GET /": "API info",
            "GET /health": "Health check",
            "POST /enhance": "Enhance image",
            "GET /demo": "Interactive demo"
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "nafnet": MODEL is not None,
        "device": str(DEVICE)
    }


@app.post("/enhance")
async def enhance(image: UploadFile = File(...)):
    """Enhance image."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    start = time.time()
    result, timings = process_image(img)
    total_time = (time.time() - start) * 1000
    
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    buf.seek(0)
    
    print(f"Enhanced {image.filename} | Total: {total_time:.0f}ms")
    
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={
            "X-Total-Time-Ms": str(round(total_time, 1)),
        }
    )


@app.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Simple demo UI - shows enhanced image only."""
    
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AutoHDR Professional</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        
        h1 { 
            text-align: center; 
            margin-bottom: 10px;
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        
        .upload-area {
            border: 2px dashed #00d9ff;
            border-radius: 10px;
            padding: 50px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { 
            background: rgba(0, 217, 255, 0.1);
        }
        .upload-area.has-image {
            border-style: solid;
            background: rgba(0, 255, 136, 0.1);
            border-color: #00ff88;
        }
        
        .btn {
            background: linear-gradient(135deg, #00d9ff, #00ff88);
            color: #1a1a2e;
            border: none;
            padding: 15px 40px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
            transition: all 0.3s;
            display: block;
            width: 100%;
            margin-top: 20px;
        }
        .btn:hover { 
            transform: scale(1.02);
            box-shadow: 0 5px 20px rgba(0, 217, 255, 0.4);
        }
        .btn:disabled { 
            opacity: 0.5; 
            cursor: not-allowed;
            transform: none;
        }
        
        #loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        .spinner {
            border: 3px solid #333;
            border-top: 3px solid #00d9ff;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .result-image {
            width: 100%;
            border-radius: 10px;
            display: none;
        }
        
        .result-label {
            text-align: center;
            margin-top: 15px;
            padding: 8px 20px;
            background: rgba(0, 255, 136, 0.2);
            border-radius: 20px;
            display: inline-block;
            font-size: 14px;
            color: #00ff88;
        }
        
        .result-container {
            text-align: center;
        }
        
        .stats {
            text-align: center;
            margin-top: 15px;
            color: #888;
            font-size: 14px;
        }
        
        .download-btn {
            background: linear-gradient(135deg, #00ff88, #00d9ff);
            margin-top: 15px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 AutoHDR Professional</h1>
        <p class="subtitle">AI-Powered Real Estate Photo Enhancement</p>
        
        <div class="card">
            <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                <p id="uploadText">📷 Click to upload a real estate photo</p>
                <p style="color: #666; margin-top: 10px; font-size: 14px;">Automatic professional enhancement</p>
                <input type="file" id="fileInput" accept="image/*" style="display:none">
            </div>
            
            <button class="btn" id="enhanceBtn" disabled>✨ Enhance Photo</button>
        </div>
        
        <div id="loading">
            <div class="spinner"></div>
            <p>Enhancing your photo...</p>
        </div>
        
        <div class="card" id="resultCard" style="display:none;">
            <div class="result-container">
                <img id="resultImg" class="result-image" src="" alt="Enhanced">
                <div class="result-label">✨ Enhanced Image</div>
            </div>
            
            <div class="stats" id="stats"></div>
            
            <button class="btn download-btn" id="downloadBtn">⬇️ Download Enhanced Photo</button>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        let enhancedBlob = null;
        
        document.getElementById('fileInput').addEventListener('change', (e) => {
            selectedFile = e.target.files[0];
            if (selectedFile) {
                document.getElementById('uploadText').textContent = '✅ ' + selectedFile.name;
                document.getElementById('uploadArea').classList.add('has-image');
                document.getElementById('enhanceBtn').disabled = false;
            }
        });
        
        document.getElementById('enhanceBtn').addEventListener('click', async () => {
            if (!selectedFile) return;
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultCard').style.display = 'none';
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/enhance', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Enhancement failed');
                
                enhancedBlob = await response.blob();
                const resultUrl = URL.createObjectURL(enhancedBlob);
                
                document.getElementById('resultImg').src = resultUrl;
                document.getElementById('resultImg').style.display = 'block';
                
                document.getElementById('loading').style.display = 'none';
                document.getElementById('resultCard').style.display = 'block';
                document.getElementById('downloadBtn').style.display = 'block';
                
                const totalTime = response.headers.get('X-Total-Time-Ms');
                document.getElementById('stats').innerHTML = `⏱️ Processing time: ${totalTime}ms`;
                
            } catch (error) {
                alert('Error: ' + error.message);
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        document.getElementById('downloadBtn').addEventListener('click', () => {
            if (enhancedBlob) {
                const url = URL.createObjectURL(enhancedBlob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'enhanced_' + (selectedFile?.name || 'image.png');
                a.click();
            }
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("  AutoHDR Professional Server")
    print("=" * 60)
    print("\nStarting server on http://0.0.0.0:8000")
    print("Open: http://localhost:8000/demo")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)