# DVR Guard

A lightweight person detection system for Hikvision DVRs using YOLOv8n (ONNX) and Telegram alerts.

## Overview

DVR Guard monitors RTSP sub-streams from Hikvision DVRs, detects persons in real-time using YOLOv8n, and sends annotated snapshot alerts via Telegram. Includes a web dashboard for viewing camera status, recent alerts, and snapshot history.

## Key Features

- **Person Detection**: YOLOv8n ONNX model with hardware acceleration via ONNX Runtime
- **Motion Gating**: Contour-based motion detection reduces unnecessary inference calls
- **Multi-Camera Support**: Monitor multiple cameras concurrently with per-camera settings
- **Telegram Alerts**: Automatic notifications with annotated snapshots when persons detected
- **Web Dashboard**: Flask-based interface at port 8080 showing camera status and alert history
- **Configurable**: Per-camera motion thresholds, detection confidence, and alert cooldowns

## Technology Stack

- **Python 3**
- **OpenCV** - Frame capture and image processing
- **ONNX Runtime** - Optimized YOLOv8n inference
- **Flask** - Web dashboard
- **python-telegram-bot** - Alert notifications
- **PyYAML** - Configuration management

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <your-repo-url>
   cd dvr-guard
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure**:
   ```bash
   cp config.example.yaml config.yaml
   # Edit config.yaml with your DVR RTSP URLs, Telegram bot token, etc.
   ```

3. **Add model**:
   - Place `yolo26n.onnx` in the `models/` directory
   - Or export from `yolo26n.pt` using YOLOv8 export tools

4. **Run**:
   ```bash
   python src/main.py
   ```

5. **Access dashboard**: Open `http://localhost:8080`

## Configuration

Edit `config.yaml` (based on `config.example.yaml`):
- **Cameras**: RTSP URLs (use sub-stream channels like 102, 202, 302, 402 for 640p)
- **Detection**: Confidence threshold, model path
- **Motion**: Contour area threshold, pixel threshold
- **Telegram**: Bot token and chat ID
- **Dashboard**: Port, authentication, public access settings


## Notes

- Uses **sub-streams only** (channels 102/202/302/402) for efficient 640p processing
- Single inference thread for YOLO calls (not parallelized)
- Motion gate is mandatory - YOLO is only called when motion is detected
- Model must be exported with `nms=True` (output shape: `(1, 300, 6)`)

*Vibecoded with [pi.dev](pi.dev).*
