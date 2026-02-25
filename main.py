# main.py

# siyolo -- A Simple YOLO Inference Server
# A drop in replacement for DeepStack AI and/or CodeProject.AI running YOLO models
#
# https://github.com/nemesis2/siyolo
# Released under the MIT License, see included LICENSE
# Last Updated: 2026-02-24 -- By nemesis2

VERSION = "v1.5" 

# Server configuration
import os
SERVER_LISTEN = os.environ.get("SERVER_LISTEN", "127.0.0.1")                             # ip address(es) to bind to (127.0.0.1)
SERVER_PORT = int(os.environ.get("SERVER_PORT", 32168))                                  # listening port (32168/5000)
SERVER_MODEL = os.environ.get("SERVER_MODEL", "yolov8x.pt")                              # YOLO model to use
MINIMUM_CONFIDENCE = float(os.environ.get("MINIMUM_CONFIDENCE", 0.65))                   # default confidence if not set in post (0.65)

# Boolean check: returns True only if the env var is literally "True" (case-insensitive)   
YOLO_VERBOSE = os.environ.get("YOLO_VERBOSE", "False").lower() == "true"                 # Show YOLO output/other errors, True or False (False, note caps)
UVICORN_LOG = os.environ.get("UVICORN_LOG", "warning")                                   # uvicorn log level (warning)
CUDA_HALF = os.environ.get("CUDA_HALF", "True").lower() == "true"                        # Use FP16 if on CUDA (True, Set to False if compute < 7)

IMG_SZ_X = int(os.environ.get("IMG_SZ_X", 640))                                         # image size to inference (default 640)
IMG_SZ_Y = int(os.environ.get("IMG_SZ_Y", 480))                                          # 480; must be a multiple of 32!
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 10 * 1024 * 1024))               # 10MB Max image size

import time
import json
import base64
import asyncio
import logging
import platform
import threading
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import sys
import torch
import numpy as np
from ultralytics import YOLO
from fastapi.responses import ORJSONResponse
from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException


# Configure logging
# If running in a terminal, show time. If piped to syslog, keep it clean.
log_format = '%(asctime)s [%(levelname)s] %(message)s' if sys.stdin.isatty() else '[%(levelname)s] %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("siyolo")


def detect_cuda():
    """Detect CUDA and FP16 capability; return device, use_fp16, major, minor"""
    device = "cpu"
    use_fp16 = False
    major = minor = -1

    if torch.cuda.is_available():
        device = "cuda"
        major, minor = torch.cuda.get_device_capability()
        if CUDA_HALF:  # Run FP16 requirement and validation checks
            if major >= 7:  # Volta or newer
                try:
                    cuda_device = torch.device("cuda")
                    x = torch.randn(16, 16, device=cuda_device, dtype=torch.float16)
                    y = torch.matmul(x, x)
                    _ = y.sum().item()
                    use_fp16 = True
                except Exception:
                    logger.info("FP16 requested but failed validation test — using FP32")
                    use_fp16 = False
                if major >= 8:  # Ampere or newer
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
            else:
                logger.info("FP16 requested but compute capability < 7.0 — using FP32")
    return device, use_fp16, major, minor


def get_cpu_name():
    """Return CPU model name"""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    return line.split(":")[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def decode_image(img_data):
    """Synchronous image decoding for thread offloading"""
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def run_inference_sync(app, img, min_confidence):
    """Synchronous function to run inference and parse boxes to avoid blocking the event loop."""
    start_time = time.time()
    
    # Run model
    with app.state.inference_lock:
        with torch.inference_mode():
            results = app.state.model(img, imgsz=(IMG_SZ_X, IMG_SZ_Y), conf=min_confidence, half=app.state.use_fp16, verbose=YOLO_VERBOSE, stream=False)[0]
    inference_ms = int((time.time() - start_time) * 1000)

    predictions = []
    
    # Fast vectorized filtering using the raw tensor data [x1, y1, x2, y2, conf, cls]
    boxes = results.boxes

    #if boxes is None or boxes.data is None or len(boxes.data) == 0:
    if boxes is None or boxes.data.shape[0] == 0:
        return predictions, inference_ms

    data = boxes.data.cpu().numpy()  # Only copy what YOLO already filtered
    names = app.state.names
    names_count = len(names)
    h, w = img.shape[:2]

    for row in data:
        x_min, y_min, x_max, y_max, conf, cls_id = row  # Note: Assuming column order from YOLO model
    
        # Boundary checks
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(w - 1, int(x_max)), min(h - 1, int(y_max))
    
        if x_max <= x_min or y_max <= y_min:
            continue
        
        cls_id = int(cls_id)
        if 0 <= cls_id < names_count:
            predictions.append({
                "confidence": round(float(conf), 4),
                "label": names[cls_id],
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max
            })

    return predictions, inference_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan hook — worker-only startup/shutdown"""
    logger.info("Simple YOLO inference worker starting")

    device, use_fp16, major, minor = detect_cuda()

    # Save state to the app instead of globals
    app.state.device = device
    app.state.use_fp16 = use_fp16

    # Load YOLO model
    BASE_DIR = Path(__file__).resolve().parent
    MODEL_DIR = BASE_DIR / "models"
    MODEL_NAME = SERVER_MODEL
    app.state.model_name = MODEL_NAME
    MODEL_PATH = MODEL_DIR / MODEL_NAME

    try:
        app.state.model = YOLO(str(MODEL_PATH))
        app.state.model.to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    app.state.inference_lock = threading.Lock()
    app.state.names = app.state.model.names
    
    # Startup logging
    cuda_ver = torch.version.cuda or "N/A"
    logger.info(f"Using Torch version: {torch.__version__} (CUDA: {cuda_ver})")
    fp = "16" if use_fp16 else "32"

    if device == "cuda":
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Running model {MODEL_NAME} @ {IMG_SZ_X}x{IMG_SZ_Y}px on {device.upper()} device 0 "
              f"({device_name}, compute {major}.{minor}/sm_{major}{minor}) using FP{fp} precision")

        # Warm-up
        dummy = np.zeros((IMG_SZ_X, IMG_SZ_Y, 3), dtype=np.uint8)
        with torch.inference_mode():
            for _ in range(2):
                app.state.model(dummy, imgsz=(IMG_SZ_X, IMG_SZ_Y), half=use_fp16, verbose=False)
        torch.cuda.synchronize()

        props = torch.cuda.get_device_properties(0)
        mem_allocated_device_0 = torch.cuda.memory_allocated(0)
        mem_allocated_reserved_0 = torch.cuda.memory_reserved(0)
        logger.info(f"Total GPU Memory: {props.total_memory / 1024 / 1024:.2f} MiB, "
              f"Allocated: {mem_allocated_device_0 / 1024 / 1024:.2f} MiB, "
              f"Reserved: {mem_allocated_reserved_0 / 1024 / 1024:.2f} MiB")
        torch.backends.cudnn.benchmark = True
    else:
        device_name = get_cpu_name()
        logger.info(f"Running model {MODEL_NAME} @ {IMG_SZ_X}x{IMG_SZ_Y}px on {device.upper()} ({device_name}), using FP{fp} precision")
        dummy = np.zeros((IMG_SZ_X, IMG_SZ_Y, 3), dtype=np.uint8)
        with torch.inference_mode():
            app.state.model(dummy, imgsz=(IMG_SZ_X, IMG_SZ_Y), verbose=False)

    logger.info(f"Simple YOLO inference server {VERSION} ready and listening on {SERVER_LISTEN}:{SERVER_PORT}")

    yield

    # Shutdown
    logger.info("Simple YOLO inference worker exiting")


# FastAPI app
app = FastAPI(title="Simple YOLO Inference Server", version=VERSION, lifespan=lifespan)


# health ping
@app.get("/health")
async def health_check():
    return {"status": "ok", "version": VERSION}


@app.post("/v1/vision/detection")
async def detect(
    request: Request, 
    image: UploadFile = File(None),
    min_confidence: float = Form(MINIMUM_CONFIDENCE)
):
    """DeepStack-compatible detection endpoint"""
    if not hasattr(request.app.state, 'model') or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not (0.0 <= min_confidence <= 1.0):
        logger.error(f"min_confidence must be between 0.0 and 1.0")
        raise HTTPException(status_code=400, detail="min_confidence must be between 0.0 and 1.0")

    raw_data = None

    # Handle UploadFile
    if image is not None:
        try:
            raw_data = await image.read()
            if not raw_data or len(raw_data) > MAX_IMAGE_BYTES:
                logger.error(f"Invalid or oversized image")
                raise HTTPException(status_code=400, detail="Invalid or oversized image")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"File read error: {e}")
            raise HTTPException(status_code=400, detail=f"File read error: {e}")

    # Handle JSON base64
    elif "application/json" in request.headers.get("content-type", ""):
        try:
            body_json = await request.json()
            if "image" not in body_json:
                raise ValueError("Missing 'image' in JSON")
            
            try:
                min_confidence = float(body_json.get("min_confidence", min_confidence))
            except (TypeError, ValueError):
                logger.error(f"min_confidence must be between 0.0 and 1.0")
                raise HTTPException(status_code=400, detail="Invalid min_confidence")
                
            if not (0.0 <= min_confidence <= 1.0):
                logger.error(f"min_confidence must be between 0.0 and 1.0")
                raise HTTPException(status_code=400, detail="min_confidence must be between 0.0 and 1.0")
            img_b64 = body_json["image"]
            if img_b64.startswith("data:"):
                img_b64 = img_b64.split(",", 1)[1]
            raw_data = base64.b64decode(img_b64, validate=True)
            if not raw_data or len(raw_data) > MAX_IMAGE_BYTES:
                logger.error(f"Invalid or oversized image")
                raise HTTPException(status_code=400, detail="Invalid or oversized image")
                
        except HTTPException:
            raise
            
        except Exception as e:
            logger.error(f"JSON decode error: {e}")
            raise HTTPException(status_code=400, detail=f"JSON decode error: {e}")
    else:
        logger.error(f"No image provided / Invalid Content-Type")
        raise HTTPException(status_code=400, detail="No image provided / Invalid Content-Type")

    # Decode image in a separate thread to prevent event loop blocking
    try:
        img = await asyncio.to_thread(decode_image, raw_data)
        if img is None:
            raise ValueError("Cannot decode image format")
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        raise HTTPException(status_code=400, detail=f"Image decode failed: {e}")

    # Run inference in a separate thread
    predictions, inference_ms = await asyncio.to_thread(
        run_inference_sync, request.app, img, min_confidence
    )

    response = {
        "success": True,
        "count": len(predictions),
        "inferenceMs": inference_ms,
        "predictions": predictions
    }
    
    return ORJSONResponse(response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=SERVER_LISTEN, port=SERVER_PORT, log_level=UVICORN_LOG.lower())
