# main.py

# SiYolo
#
# A Simple YOLO Inference Server
# Made to be a drop in (more or less) replacement for Deepstacks and/or CodeProject.AI.
# Released under an opensource license (GPL? MPL? ...?)
# By Nemesis][ -- nemesis2@gmail.com
#
#

# Server configuration
SERVER_LISTEN = "127.0.0.1"		# ip address(es) to bind to
SERVER_PORT = 32168				# listening port (32168)
SERVER_MODEL = "yolov8x.pt"		# YOLO model to use if no environment variable set
MINIMUM_CONFIDENCE = 0.65		# default confidence if not set in post
YOLO_VERBOSE = False			# Show YOLO output and other errors, True or False (note caps)
UVICORN_LOG = "warning"			# uvicorn log level
CUDA_HALF = False				# Use FP16 if on CUDA (Keep off for Maxwell era)
VERSION = "v1.0-dev"			# Version

MAX_IMAGE_BYTES = 10 * 1024 * 1024	# 10MB Max image size

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from pathlib import Path
from ultralytics import YOLO
from contextlib import asynccontextmanager

import torch
import numpy as np
import cv2
import base64
import os
import time
import math

# CUDA / GPU Detection
if torch.cuda.is_available():
	device = "cuda"
else:
	device = "cpu"
	CUDA_HALF = False  # CPU must stay FP32

# Load YOLO model
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_NAME = os.environ.get("YOLO_MODEL", SERVER_MODEL)
MODEL_PATH = MODEL_DIR / MODEL_NAME

try:
	model = YOLO(str(MODEL_PATH))
	model.to(device)
except Exception as e:
	raise RuntimeError(f"Failed to load YOLO model: {e}")

# Return CPU type if no CUDA
def get_cpu_name():
	# Try for Linux/Unix
	try:
		with open("/proc/cpuinfo", "r") as f:
			for line in f:
				if "model name" in line:
					return line.split(":")[1].strip()
	except Exception:
		pass

	# Other OSes
	import platform	   
	return platform.processor() or platform.machine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
	print(f"Using Torch version: {torch.__version__} (CUDA: {torch.version.cuda})")
	if device == "cuda":
		device_name = torch.cuda.get_device_name(0)
		mem_allocated_device_0 = torch.cuda.memory_allocated(0)
		mem_allocated_reserved_0 = torch.cuda.memory_reserved(0)
		print(f"Torch memory allocated: {mem_allocated_device_0 / 1024 / 1024:.2f} MiB, "
			  f"Reserved: {mem_allocated_reserved_0 / 1024 / 1024:.2f} MiB")
	else:
		device_name = get_cpu_name()
	print(f"Running model {MODEL_NAME} on {device.upper()} ({device_name})")
	print(f"Simple YOLO inference server {VERSION} ready and listening on {SERVER_LISTEN}:{SERVER_PORT}")

	yield  # here FastAPI runs

	# shutdown
	print("Simple YOLO inference server exiting...")
	
# API Endpoint
app = FastAPI(title="Simple YOLO Inference Server", version=VERSION, lifespan=lifespan)
	
@app.post("/v1/vision/detection")  # Mimic deepstacks, codeproject.ai path
async def detect(
	request: Request, 
	image: UploadFile = File(None), 
	min_confidence: float = Form(MINIMUM_CONFIDENCE)
):
	img = None
	
	# Standard multipart upload
	if image is not None:
		if not (0.0 <= min_confidence <= 1.0):
			raise HTTPException(
				status_code=400,
				detail="min_confidence must be between 0.0 and 1.0"
			)			
		try:
			# Read image
			data = await image.read()
			if not data or len(data) > MAX_IMAGE_BYTES:
				raise HTTPException(status_code=400, detail="Invalid or oversized image")
			nparr = np.frombuffer(data, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			if img is None:
				raise ValueError("Cannot decode uploaded file")

		except HTTPException:
			raise			 

		except Exception as e:
			if YOLO_VERBOSE:
				print(f"[ERROR] Multipart decode error: {e}")
			raise HTTPException(status_code=400, detail=f"Multipart decode error: {e}")

	# JSON base64 fallback for future usage...?
	elif "application/json" in request.headers.get("content-type", ""):
		try:
			body_json = await request.json()
			if "image" not in body_json:
				raise ValueError("Missing 'image' in JSON")
			# Honor client's min_confidence
			min_confidence = float(body_json.get("min_confidence", min_confidence))
			if not (0.0 <= min_confidence <= 1.0):
				raise HTTPException(
					status_code=400,
					detail="min_confidence must be between 0.0 and 1.0"
				)			 
			img_data = base64.b64decode(body_json["image"], validate=True)
			if not img_data or len(img_data) > MAX_IMAGE_BYTES:
				raise HTTPException(status_code=400, detail="Invalid or oversized image")			 
			nparr = np.frombuffer(img_data, np.uint8)
			img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
			if img is None:
				raise ValueError("Cannot decode base64 image")
			
		except HTTPException:
			raise			 

		except Exception as e:
			if YOLO_VERBOSE:
				print(f"[ERROR] JSON decode error: {e}")
			raise HTTPException(status_code=400, detail=f"JSON decode error: {e}")

	# No image posted...?
	else:
		if YOLO_VERBOSE:
			print(f"[ERROR] No image provided / Invalid Content-Type")
		raise HTTPException(status_code=400, detail="No image provided / Invalid Content-Type")
		
	if img is None:
		if YOLO_VERBOSE:
			print(f"[ERROR] Image decode failed")		 
		raise HTTPException(status_code=400, detail="Image decode failed")		  

	# Run the YOLO inference (for newer ultralytics use: verbose->show=False)
	start_time = time.time()
	with torch.inference_mode():
		results = model(img, imgsz=640, half=CUDA_HALF, verbose=YOLO_VERBOSE)[0]
	inference_ms = int((time.time() - start_time) * 1000)

	# Filter detections by confidence
	predictions = []
	h, w = img.shape[:2]
	
	for box in results.boxes:
		conf = float(box.conf[0])
		
		# YOLO occasionally emits NaN confidences (FP16 + Maxwell)
		# So we add sanity/bounds checks:
		if not math.isfinite(conf) or conf < 0.001:
			continue
		if conf < min_confidence or conf > 1.0:
			continue

		# convert and check bounding box coords
		x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
		if not all(map(math.isfinite, (x_min, y_min, x_max, y_max))):
			continue
		if x_max <= x_min or y_max <= y_min:
			continue
		
		# verify they are within the size of the image itself		 
		if x_min < 0 or y_min < 0 or x_max > w or y_max > h:
			continue
		
	        # verify class id is sane        
		cls_id = int(box.cls[0])
		if cls_id < 0 or cls_id >= len(model.names):
			continue

		# we got this far, add it to the return list
		predictions.append({
			"confidence": round(conf, 4),
			"label": model.names[cls_id],
			"x_min": int(x_min),
			"y_min": int(y_min),
			"x_max": int(x_max),
			"y_max": int(y_max)
		})
		
	response = {
		"success": True,
		"predictions": predictions,
		"count": len(predictions),
		"inferenceMs": inference_ms
	}

	if YOLO_VERBOSE:
		print(f"[INFO] Detections={len(predictions)}, min_conf={min_confidence}, time={inference_ms}ms")

	return response

if __name__ == "__main__":
	# "startup"
	print("Simple YOLO inference server starting...")
	import uvicorn
	uvicorn.run("main:app", host=SERVER_LISTEN, port=SERVER_PORT, log_level=UVICORN_LOG)
