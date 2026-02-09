# siyolo: A Simple YOLO Inference Server 

## Overview

A lightweight, locally-hosted YOLO server written in Python using FastAPI.

Supports CPU, Maxwell GPUs (GTX 970/980), modern NVIDIA GPUs (RTX 30/40), and Windows/Linux environments.

Made as a drop-in replacement for Deepstacks or Codeproject.AI inference servers.

## Linux Installation & Setup Guide

1)  Directory Setup

    Create the server and model directories:
```
sudo mkdir -p /opt/siyolo/models
cd /opt/siyolo
```


2) Python Virtual Environment

  ⚠ Linux: Use Python 3.10 for older CPUs/Maxwell GPUs. Windows can use Python 3.10–3.12.
```   
python3 -m venv ./venv
source ./venv/bin/activate   # Linux/macOS
pip install --upgrade pip
```

3) Install Requirements

  a) Modern GPUs / New CPUs (CUDA 12.1+)
  
  ```
    pip install torch torchvision numpy ultralytics fastapi uvicorn opencv-python python-multipart --index-url https://download.pytorch.org/whl/cu121
  ```

b) Maxwell GPUs (GTX 970 / 980, CUDA 11.7)

```
  pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 numpy ultralytics==8.0.200 fastapi uvicorn opencv-python python-multipart --index-url https://download.pytorch.org/whl/cu117
```

c) Older CPUs (no AVX2 / AVX512)

```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 numpy==1.24.4 ultralytics==8.0.200 fastapi uvicorn opencv-python python-multipart --index-url https://download.pytorch.org/whl/cu117
```

4) Extract Server Files

If you have a tarball:

tar -xzvf siyolo.tar.gz -C /opt/siyolo

5) Create System User (Linux Only)

```
sudo useradd -r siyolo
sudo chown -R siyolo:siyolo /opt/siyolo
```

Server runs under a dedicated user for security.

6) Configure systemd Service (Linux Only)

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
Environment="TMPDIR=/dev/shm"
Environment="PYTHONUNBUFFERED=1"
Environment="YOLO_MODEL=yolov8x.pt"
Environment="YOLO_VERBOSE=False"

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

Enable and start:

```
sudo ln -s /opt/siyolo/siyolo.service /etc/systemd/system/siyolo.service
sudo systemctl daemon-reload
sudo systemctl enable siyolo
sudo systemctl start siyolo
sudo systemctl status siyolo
```

7) Testing the Server (Linux & Windows)

Activate venv and run manually first:

```
source /opt/siyolo/venv/bin/activate  # Linux
python3.10 main.py
```
Expected output:

Starting Simple YOLO server v1.0, listening on 0.0.0.0:32168

Torch version: 1.13.1+cu117 (CUDA: 11.7)

Running model yolov8x.pt on CUDA (NVIDIA GeForce GTX 970)


    • Use curl or your client to POST images for inference.
    • Supports multipart/form-data and application/json (base64).
    • Honor min_confidence from client requests.


Notes & Tips
    • Models: Place YOLO .pt files in /opt/siyolo/models/.
    
    • CPU Display: Falls back to system CPU if no CUDA.
    
    • FP16: Use half=True on CUDA for lower VRAM usage.
    
    • Threading: OMP_NUM_THREADS=1 is safe for dual-core; increase for multi-core CPUs.
    
    • VRAM: Large models (yolov8x-seg.pt) may require >3–4GB. Consider smaller models (yolov8n, yolov8m) for 4GB GPUs.
    
    • Debug Logs: Controlled via YOLO_VERBOSE=True/False.

