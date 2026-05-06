# DVR Guard — Project Context for Pi

## What this project does
Detects persons in RTSP sub-streams from a Hikvision DS-7104HQHI-K1 DVR.
Uses YOLO26n (ONNX) for inference. Sends Telegram alerts with annotated snapshots.
Runs a local Flask dashboard on port 8080 for camera status and recent alerts.


## Key constraints
- Sub-streams ONLY: use channels 102, 202, 302, 402 (640p) not 101/201/301/401 (1080p)
- Single ONNX inference thread — do NOT parallelize YOLO calls
- Motion gate is mandatory — never call YOLO on every frame
- Motion gate uses contour area (NOT diff.mean) — required for IR/night camera noise
- Per-camera motion overrides: each camera in config.yaml can have its own `motion:` block
  that overrides the global motion settings. camera.py merges them at runtime.
- Notifier runs in its own thread with a queue — never block the inference loop on network calls
- Per-camera cooldown: default 60s between alerts for same camera (configurable per camera in config.yaml)

## Module responsibilities (under src/)
- state.py: SharedState class — thread-safe, holds camera status + recent detections
- camera.py: CameraThread — one thread per camera, frame grab + motion gate (uses merged motion config)
- detector.py: PersonDetector — YOLO26n ONNX wrapper + NMS
- notifier.py: TelegramNotifier — own queue/thread, non-blocking send_alert()
- dashboard.py: Flask app — serves status and snapshots, reads SharedState
- main.py: wires everything together

## Commands
- Start: python src/main.py
- Test detector: python src/detector.py --image test.jpg
- Test single camera: python src/camera.py --cam 0 --debug
- Dashboard: http://localhost:8080


## detector.py — Output Format (CONFIRMED)

YOLO26n was exported WITH `nms=True`.
Confirmed output shape: `(1, 300, 6)`
  - 300 = max detections, zero-padded
  - 6   = [x1, y1, x2, y2, confidence, class_id]
  - Coordinates are in letterboxed input pixel space (0..640)
  - NMS is already applied — do NOT add cv2.dnn.NMSBoxes

Do NOT change postprocess to handle [1, 84, 8400] — that shape is for exports WITHOUT nms=True, which is not what this model uses (`nms_iou_threshold` in config.yaml is inert)
