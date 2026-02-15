# siyolo: A Simple YOLO Inference Server 

## Overview

A lightweight, locally-hosted Ultralytics YOLO inference server written in Python using FastAPI.

siyolo was designed as a drop-in replacement for DeepStack and CodeProject.AI,  
providing fast local object detection without heavyweight AI frameworks or runtime environments.

Simple, fast and locally-hosted. No Docker, no .NET needed, just Python.  
A single-process, LAN-focused inference server for local object detection workloads.

---

* Suitable for NVR and home automation
* Minimal dependencies (no .NET or Docker required)
* Fast startup with CUDA warm-up
* Automatic FP16 detection (Volta+ GPUs)
* Supports CPU and CUDA (Linux and Windows)
* Defensive filtering of NaN / invalid detections
* DeepStack-compatible REST API
* Optimized for low-memory environments
* Ultralytics YOLO backend + FastAPI
* Supports multipart/form-data and application/json (base64)

    
---

## Linux Installation & Setup Guide

### Install from git

```
cd /opt
sudo git clone https://github.com/nemesis2/siyolo.git
```

### Directory Setup

Create the model directory:
```
sudo mkdir -p /opt/siyolo/models
cd /opt/siyolo
```


### Install Python Virtual Environment

⚠ Linux: Use Python 3.10 for older CPUs/Maxwell GPUs. Windows can use Python 3.10–3.12.
```   
python3 -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip
```

### Python Package Requirements

⚠ Linux: The correct combination may vary depending on CPU/GPU and distribution.

Modern GPUs / New CPUs (CUDA 12.1+)
  
```
pip install torch torchvision numpy ultralytics fastapi uvicorn opencv-python python-multipart \
--index-url https://download.pytorch.org/whl/cu121
```

Maxwell GPUs (GTX 970 / 980, CUDA 11.7)

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 numpy ultralytics==8.0.200 fastapi uvicorn \
opencv-python python-multipart --index-url https://download.pytorch.org/whl/cu117
```

Older CPUs (no AVX2 / AVX512)

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 numpy==1.24.4 ultralytics==8.0.200 \
fastapi uvicorn opencv-python python-multipart --index-url https://download.pytorch.org/whl/cu117
```

### Create System User

```
sudo useradd -r siyolo
sudo chown -R siyolo:siyolo /opt/siyolo
```

### Configure systemd Service 

Create /opt/siyolo/siyolo.service:

```
[Unit]
Description=A Simple YOLO Inference Server
After=network.target

[Service]
Type=simple
User=siyolo
Group=siyolo
WorkingDirectory=/opt/siyolo
ExecStart=/opt/siyolo/venv/bin/python3.10 -u /opt/siyolo/main.py
Restart=always
RestartSec=5

# Environment
Environment=MPLCONFIGDIR=/opt/siyolo/.config/matplotlib
Environment=YOLO_CONFIG_DIR=/opt/siyolo/.config/Ultralytics
Environment=TMPDIR=/dev/shm
Environment=PYTHONUNBUFFERED=1
Environment=YOLO_MODEL=yolov8x.pt
Environment=YOLO_VERBOSE=False

# Prevent CPU oversubscription
Environment=OMP_NUM_THREADS=1
Environment=OPENBLAS_NUM_THREADS=1
Environment=MKL_NUM_THREADS=1
Environment=NUMEXPR_NUM_THREADS=1

# Optional limits
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=true

[Install]
WantedBy=multi-user.target
```

### Testing the Server

Activate venv and run manually first:

```
source /opt/siyolo/venv/bin/activate 
python3.10 main.py
```
Expected output should be similar to:

```
Simple YOLO inference server starting...
Using Torch version: 1.13.1+cu117 (CUDA: 11.7)
Running model yolov8x.pt on CUDA device 0 (NVIDIA GeForce GTX 970, compute 5.2/sm_52) using FP32 precision
Total GPU Memory: 4036.75 MiB, Allocated: 534.74 MiB, Reserved: 706.00 MiB
Simple YOLO inference server v1.2-dev ready and listening on 127.0.0.1:32168
```

### Verify Inferencing

Test image: https://deepstack.readthedocs.io/en/latest/_images/family-and-dog.jpg

```
curl -s -X POST -F 'image=@family-and-dog.jpg' 'http://127.0.0.1:32168/v1/vision/detection' | jq '.'
```

Should return something like:
 
```jsonc
{
  "success": true,
  "count": 3,
  "inferenceMs": 74,  
  "predictions": [
    {
      "confidence": 0.9401,
      "label": "person",
      "x_min": 294,
      "y_min": 85,
      "x_max": 442,
      "y_max": 519
    },
    {
      "confidence": 0.9371,
      "label": "dog",
      "x_min": 650,
      "y_min": 344,
      "x_max": 793,
      "y_max": 540
    },
    {
      "confidence": 0.9249,
      "label": "person",
      "x_min": 443,
      "y_min": 113,
      "x_max": 601,
      "y_max": 523
    }
  ]
}
```


### Enable and start:

```
sudo ln -s /opt/siyolo/siyolo.service /etc/systemd/system/siyolo.service
sudo systemctl daemon-reload
sudo systemctl enable siyolo
sudo systemctl start siyolo
sudo systemctl status siyolo
```

---

## Notes

⚠ This server does not implement authentication, rate limiting or request throttling. Do not expose directly to the public Internet. ⚠ 

* Default inference size: 640x640 (configurable in main.py)
* Place YOLO .pt files in /opt/siyolo/models/; if missing system will attempt to download automatically
* Fails over to system CPU if no CUDA present
* Additional debugging controlled via YOLO_VERBOSE=True/False

