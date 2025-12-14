# AutoHDR Challenge — NAFNet-Based Real-Estate Photo Enhancement

AutoHDR is a **single-image, paired-supervision** photo enhancement pipeline that learns to transform a standard (often flat / low dynamic range) real-estate photo into a more **HDR-like** image: lifted interiors, controlled highlights, and more consistent tone/color.

This repository includes:
- Data preprocessing into paired train/val folders
- Training + fine-tuning of a NAFNet-based image-to-image model
- Batch inference and evaluation (PSNR/SSIM)
- A simple **web upload demo** (FastAPI) that returns the enhanced image
- Optional model export to **ONNX** and a TensorRT export path

---

## Demo Video

YouTube: https://youtu.be/w7kcbpZhnhs

## Contents
- [Demo Video](#demo-video)
- [Quickstart (Web Demo)](#quickstart-web-demo)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data Format & Preprocessing](#data-format--preprocessing)
- [Training](#training)
- [Fine-tuning](#fine-tuning)
- [Batch Inference](#batch-inference)
- [Evaluation](#evaluation)
- [Benchmarking](#benchmarking)
- [Export (ONNX / TensorRT)](#export-onnx--tensorrt)
- [Results](#results)
- [Troubleshooting](#troubleshooting)
- [Hackathon Experiments (Parallel Models)](#hackathon-experiments-parallel-models)

---

## Quickstart (Web Demo)

The fastest way to see the project working end-to-end is the upload demo.

1) Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Ensure a checkpoint exists:
- Default expected checkpoint: `models/checkpoints/best_model.pth`
- The repo already contains checkpoints under `models/checkpoints/`.

3) Start the server:

```bash
python scripts/server_new.py
```

4) Open the demo UI:
- `http://localhost:8000/demo`

5) Upload any image (jpg/png) and download/view the enhanced PNG.

Notes:
- The demo resizes inputs to **1024×1024** for inference and then resizes back to the original resolution.
- If CUDA is available, inference uses AMP autocast for speed.

---

## Project Structure

Key folders:
- `scripts/`: all runnable entrypoints (preprocess/train/infer/eval/server/export)
- `data/raw/`: raw dataset inputs (paired)
- `data/processed/`: preprocessed PNGs for training/validation
- `models/checkpoints/`: saved checkpoints (including `best_model.pth`)
- `models/exports/`: exported models (ONNX, TensorRT engines)
- `outputs/`, `outputs_enhanced/`: sample inference outputs
- `benchmarks/`: evaluation and speed reports

---

## Setup

### Requirements
- Linux/macOS/Windows (Linux recommended for CUDA workflows)
- Python 3.10+ recommended
- Optional: NVIDIA GPU + CUDA for practical training speed

Install:

```bash
pip install -r requirements.txt
```

Dependencies are defined in `requirements.txt` (PyTorch, Pillow, scikit-image, FastAPI, etc.).

---

## Data Format & Preprocessing

### Expected raw data layout
The preprocessing script expects paired images in `data/raw/images/` with naming:

- `*_src.jpg` (input / LDR)
- `*_tar.jpg` (target / HDR-like)

Example:

```text
data/raw/images/000123_src.jpg
data/raw/images/000123_tar.jpg
```

### Preprocess into train/val

```bash
python scripts/preprocess.py
```

What preprocessing does:
- Converts pairs to PNG and resizes to **1024×1024**
- Splits into train/val (seeded) and writes `data/split_manifest.txt`
- Writes outputs to:
  - `data/processed/train/input` and `data/processed/train/output`
  - `data/processed/val/input` and `data/processed/val/output`

---

## Training

Train the base NAFNet model:

```bash
python scripts/train.py
```

Highlights (see `scripts/train.py`):
- Architecture: `NAFNet` (defined in `scripts/model.py`)
- Loss: **L1 + simplified SSIM term** for structure-preserving enhancement
- Scheduler: cosine annealing
- AMP: enabled automatically when CUDA is available

Outputs:
- Best checkpoint: `models/checkpoints/best_model.pth`
- Periodic checkpoints: `models/checkpoints/ckpt_epoch_10.pth`, etc.
- Logs: `logs/training_log.txt`

---

## Fine-tuning

Fine-tuning is an optional stage meant to improve brightness/color consistency and reported metrics.

```bash
python scripts/finetune.py
```

Notes:
- Fine-tune tries to load `models/checkpoints/best_model.pth` first.
- It saves periodic fine-tune checkpoints such as `models/checkpoints/finetune_epoch_5.pth`.
- It backs up the original best checkpoint once to `models/checkpoints/best_model_before_finetune.pth`.

---

## Batch Inference

Enhance a single image or a directory:

```bash
python scripts/inference.py -i <INPUT_IMAGE_OR_DIR> -o <OUTPUT_DIR>
```

Optional: choose a checkpoint:

```bash
python scripts/inference.py -i data/processed/val/input -o outputs_demo -c models/checkpoints/best_model.pth
```

Behavior:
- Writes `<stem>_enhanced.png` into the output directory
- Prints average runtime per image

### “Enhanced” inference (multi-res + optional sky replacement)
There is also `scripts/inference_enhanced.py`, which supports multi-resolution blending and optional sky replacement.

Important: it imports `sky_replacement` which is **not included** in this workspace, so it will fail unless you add a compatible module.

---

## Evaluation

Compute PSNR/SSIM against ground truth:

```bash
python scripts/evaluate.py -p <PREDICTIONS_DIR> -g <GROUND_TRUTH_DIR>
```

Example:

```bash
python scripts/evaluate.py -p outputs_demo -g data/processed/val/output
```

Outputs:
- Console summary
- Saved report: `benchmarks/evaluation_results.txt`

---

## Benchmarking

Run a quick performance benchmark:

```bash
python scripts/benchmark.py
```

This writes a report to `benchmarks/performance_report.txt` including FP32/FP16 timing and GPU memory (when CUDA is available).

---

## Export (ONNX / TensorRT)

### ONNX
An ONNX export is included under:
- `models/exports/nafnet.onnx`

### TensorRT (optional)
The script `scripts/export_tensorrt.py` provides a TensorRT export path.

```bash
python scripts/export_tensorrt.py
```

Notes:
- Requires CUDA and a working TensorRT installation (`trtexec` available on PATH).
- The script may also attempt Python-side TensorRT benchmarking; that requires TensorRT Python bindings and `pycuda`.

---

## Results

This repo includes example reports in `benchmarks/`:
- `benchmarks/evaluation_results.txt` contains PSNR/SSIM over the validation set.
- `benchmarks/performance_report.txt` contains inference timing/throughput and GPU memory notes.

---

## Troubleshooting

### “No checkpoint found” in the server
The server expects `models/checkpoints/best_model.pth`. If you trained a different file, either:
- copy/rename it to `best_model.pth`, or
- update the checkpoint path inside `scripts/server_new.py`.

### CUDA not detected
Training prints diagnostics and suggests a CUDA-enabled PyTorch wheel if `nvidia-smi` works but PyTorch can’t see the GPU.

### Web demo dependencies
If `fastapi`/`uvicorn` are missing, install via `pip install -r requirements.txt`.

### Sky replacement
`scripts/server_new.py` treats sky replacement as optional; it disables it if `sky_replace.py` is missing.

---

## Hackathon Experiments (Parallel Models)

During the hackathon we also **trained and compared multiple model families in parallel** to validate tradeoffs in quality, speed, and deployability.

These experiments are mentioned for completeness of the hackathon work; the **shipped pipeline in this repo is NAFNet**.

Examples of parallel approaches we explored:
- **MiraNet-style enhancement** experiments (alternative backbone variants)
- **3D LUT-based methods** (learned LUTs for lightweight color/tone transforms)
- Other fast restoration-style architectures and hybrid pipelines

If you want to publish those baselines inside this repo, a good structure is:
- `scripts/experiments/<model_name>/` with its own `train.py`, `inference.py`, and evaluation hooks
- A shared evaluation runner so all models report PSNR/SSIM and runtime in a consistent format
