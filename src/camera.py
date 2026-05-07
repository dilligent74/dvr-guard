"""
Camera thread module for DVR Guard.
One thread per camera: frame grab + motion gate + optional YOLO trigger.
"""

import cv2
import numpy as np
import time
import logging
import threading
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

from state import SharedState, CameraStatus, Detection

logger = logging.getLogger(__name__)


@dataclass
class MotionConfig:
    """Motion detection configuration."""
    enabled: bool = True
    blur_size: int = 21
    pixel_threshold: int = 25
    min_contour_area: int = 800

    @classmethod
    def from_dict(cls, d: dict) -> "MotionConfig":
        return cls(
            enabled=d.get("enabled", True),
            blur_size=d.get("blur_size", 21),
            pixel_threshold=d.get("pixel_threshold", 25),
            min_contour_area=d.get("min_contour_area", 800),
        )


class CameraThread(threading.Thread):
    """
    Thread for a single camera.
    Grabs frames from RTSP, runs motion detection gate,
    and triggers YOLO inference when motion is detected.
    """

    def __init__(
        self,
        camera_id: int,
        name: str,
        rtsp_url: str,
        shared_state: SharedState,
        motion_config: MotionConfig,
        detector=None,  # PersonDetector instance (injected)
        notifier=None,  # TelegramNotifier instance (injected)
        confidence_threshold: float = 0.45,
        cooldown_seconds: int = 60,
        snapshot_dir: str = "snapshots",
        tiered_snapshots: bool = False,
        max_snapshots_per_folder: int = 500,
        stop_event: Optional[threading.Event] = None,
    ):
        super().__init__(name=f"Camera-{name}", daemon=True)
        self.camera_id = camera_id
        self.name = name
        self.rtsp_url = rtsp_url
        self.shared_state = shared_state
        self.motion_config = motion_config
        self.detector = detector
        self.notifier = notifier
        self.confidence_threshold = confidence_threshold
        self.cooldown_seconds = cooldown_seconds
        self.snapshot_dir = snapshot_dir
        self.tiered_snapshots = tiered_snapshots
        self.max_snapshots_per_folder = max_snapshots_per_folder
        self.stop_event = stop_event or threading.Event()

        self._cap: Optional[cv2.VideoCapture] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._last_detection_time: Optional[datetime] = None
        self._last_no_detection_snapshot_time: Optional[float] = None  # Throttle no-detection snapshots to 1/sec
        self._last_yolo_time: Optional[float] = None  # Rate limit YOLO to 2 fps
        self._frame_count = 0
        self._start_time = time.time()
        self._fps = 0.0

        # Tiered snapshot folders
        self._tier_folders = {
            "confirmed_45plus": os.path.join(snapshot_dir, "confirmed_45plus"),
            "uncertain_35to45": os.path.join(snapshot_dir, "uncertain_35to45"),
            "weak_25to35": os.path.join(snapshot_dir, "weak_25to35"),
            "motion_no_detection": os.path.join(snapshot_dir, "motion_no_detection"),
        }

        # Create folders if tiered snapshots enabled
        if self.tiered_snapshots:
            for folder in self._tier_folders.values():
                os.makedirs(folder, exist_ok=True)

    def _connect(self) -> bool:
        """Connect to RTSP stream."""
        logger.info(f"[{self.name}] Connecting to {self.rtsp_url}")
        self._cap = cv2.VideoCapture(self.rtsp_url)
        if not self._cap.isOpened():
            logger.error(f"[{self.name}] Failed to open RTSP stream")
            return False
        # Set buffer size to minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logger.info(f"[{self.name}] Connected successfully")
        return True

    def _disconnect(self):
        """Disconnect from RTSP stream."""
        if self._cap:
            self._cap.release()
            self._cap = None

    def _detect_motion(self, frame: np.ndarray) -> bool:
        """
        Motion gate using contour area (NOT diff.mean).
        Returns True if significant motion is detected.
        """
        if not self.motion_config.enabled:
            return True  # Motion gate disabled → always trigger

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.motion_config.blur_size, self.motion_config.blur_size), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return False

        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(diff, self.motion_config.pixel_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self._prev_gray = gray

        # Check if any contour exceeds the minimum area
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.motion_config.min_contour_area:
                return True

        return False

    def _manage_folder_size(self, folder: str) -> None:
        """Delete oldest files in folder if count exceeds max_snapshots_per_folder."""
        try:
            # Ensure folder exists
            os.makedirs(folder, exist_ok=True)
            
            files = [os.path.join(folder, f) for f in os.listdir(folder)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(files) <= self.max_snapshots_per_folder:
                return
            # Sort by modification time (oldest first)
            files.sort(key=os.path.getmtime)
            # Delete oldest files
            excess = len(files) - self.max_snapshots_per_folder
            for old_file in files[:excess]:
                if os.path.exists(old_file):  # Check if file still exists before deleting
                    os.remove(old_file)
                    logger.debug(f"Deleted old snapshot: {old_file}")
                else:
                    logger.debug(f"File already removed, skipping: {old_file}")
        except Exception as e:
            logger.error(f"Error managing folder size for {folder}: {e}")

    def _save_tiered_snapshot(
        self,
        frame: np.ndarray,
        detections: list,
        tier: str,
    ) -> Optional[str]:
        """
        Save frame to appropriate tier folder with annotations.
        For motion_no_detection tier, save unannotated frame.
        """
        if not self.tiered_snapshots:
            return None

        folder = self._tier_folders.get(tier)
        if not folder:
            logger.error(f"Unknown tier: {tier}")
            return None

        # Manage folder size before saving
        self._manage_folder_size(folder)

        now = datetime.now()
        date_str = now.strftime("%Y%m%d")  # YYYYMMDD
        time_str = now.strftime("%H_%M_%S")  # HH_MM_SS
        conf_str = ""
        if detections:
            # Each snapshot with detections contains exactly one detection
            det = detections[0]
            conf = det["confidence"]
            conf_percent = round(conf * 100)  # Convert to percentage (e.g., 0.87 → 87)
            conf_str = f"c{conf_percent:03d}"  # Format as 3-digit string (e.g., c087)
        if conf_str:
            filename = f"{date_str}-{time_str}-{self.name}-{conf_str}.jpg"
        else:
            # For motion_no_detection, no confidence part
            filename = f"{date_str}-{time_str}-{self.name}.jpg"
        path = os.path.join(folder, filename)

        # Draw annotations if detections exist
        annotated = frame.copy()
        if detections and tier != "motion_no_detection":
            for det in detections:
                x, y, w, h = det["bbox"]
                conf = det["confidence"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f"Person {conf:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        cv2.imwrite(path, annotated)
        if tier != "motion_no_detection":           # don't spam the console
            logger.info(f"[{self.name}] Tiered snapshot saved ({tier}): {path}")
        return path

    def _check_cooldown(self) -> bool:
        """Check if cooldown period has passed since last detection."""
        if self._last_detection_time is None:
            return True
        elapsed = (datetime.now() - self._last_detection_time).total_seconds()
        return elapsed >= self.cooldown_seconds

    def _connect_with_retry(self) -> bool:
        """
        Attempt to connect with exponential backoff.
        Returns True when connected, False only if stop_event is set.
        Delays: 2s, 4s, 8s, 16s, 32s, 60s, 60s, ...
        """
        delay = 2
        attempt = 0
        while not self.stop_event.is_set():
            if attempt > 0:
                logger.info(f"[{self.name}] Reconnect attempt {attempt}, waiting {delay}s...")
                # Sleep in small increments so stop_event is checked promptly
                for _ in range(delay):
                    if self.stop_event.is_set():
                        return False
                    time.sleep(1)
            if self._connect():
                return True
            attempt += 1
            delay = min(delay * 2, 60)
        return False

    def run(self):
        """Main thread loop."""
        logger.info(f"[{self.name}] Starting camera thread")

        # Initial connection with retry (handles DVR boot race, network blips at startup)
        status = CameraStatus(camera_id=self.camera_id, name=self.name, online=False)
        self.shared_state.update_camera_status(status)

        if not self._connect_with_retry():
            return  # stop_event was set before we could connect

        status = CameraStatus(camera_id=self.camera_id, name=self.name, online=True)
        self.shared_state.update_camera_status(status)

        try:
            while not self.stop_event.is_set():
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(f"[{self.name}] Frame grab failed, reconnecting...")
                    self._disconnect()
                    status = CameraStatus(camera_id=self.camera_id, name=self.name, online=False)
                    self.shared_state.update_camera_status(status)
                    if not self._connect_with_retry():
                        break  # stop_event was set
                    status = CameraStatus(camera_id=self.camera_id, name=self.name, online=True)
                    self.shared_state.update_camera_status(status)
                    continue

                self.shared_state.touch_stream(self.name)
                self._frame_count += 1
                now = time.time()
                if now - self._start_time >= 1.0:
                    self._fps = self._frame_count / (now - self._start_time)
                    self._frame_count = 0
                    self._start_time = now

                # Update status
                status = CameraStatus(
                    camera_id=self.camera_id,
                    name=self.name,
                    online=True,
                    last_frame_time=datetime.now(),
                    fps=self._fps,
                )
                self.shared_state.update_camera_status(status)

                # Motion gate
                if not self._detect_motion(frame):
                    # we added a <max 1 fps when no motion> feature somewhere to not fill no motion folder 
                    time.sleep(0.03)  # ~30 fps throttle when no motion
                    continue

                # Rate limit YOLO inference to 2 fps (0.5s between analyses)
                current_time = time.monotonic()
                if self._last_yolo_time is not None and current_time - self._last_yolo_time < 0.5:
                    continue
                self._last_yolo_time = current_time

                # Run YOLO detection
                if self.detector is None:
                    continue

                # Use infer() to get raw outputs once, then postprocess multiple times
                outputs, scale, pad_offset, orig_shape = self.detector.infer(frame)

                # Postprocess with main confidence threshold (for Telegram alert)
                person_detections = self.detector._postprocess(
                    outputs, scale, pad_offset, orig_shape, self.confidence_threshold
                )

                # Handle tiered snapshots if enabled (no cooldown applied)
                confirmed_snapshot_path = None
                if self.tiered_snapshots:
                    # Postprocess with lower threshold (0.25) for lower tiers
                    low_detections = self.detector._postprocess(
                        outputs, scale, pad_offset, orig_shape, 0.25
                    )

                    if low_detections:
                        # Find the best (highest confidence) detection for snapshot path tracking
                        best_low = max(low_detections, key=lambda d: d["confidence"])
                        
                        # Route to appropriate tier based on confidence
                        for det in low_detections:
                            conf = det["confidence"]
                            if conf >= 0.45:
                                tier = "confirmed_45plus"
                            elif conf >= 0.35:
                                tier = "uncertain_35to45"
                            else:  # 0.25 <= conf < 0.35
                                tier = "weak_25to35"
                            path = self._save_tiered_snapshot(frame, [det], tier)
                            # Capture path if this is the best detection (for Telegram alert)
                            if det["confidence"] == best_low["confidence"] and det["bbox"] == best_low["bbox"]:
                                confirmed_snapshot_path = path
                    else:
                        # motion_no_detection: motion fired but nothing at 0.25
                        # Throttle: save only one per second
                        current_time = time.monotonic()
                        if (self._last_no_detection_snapshot_time is None or
                            current_time - self._last_no_detection_snapshot_time >= 1.0):
                            self._save_tiered_snapshot(frame, [], "motion_no_detection")
                            self._last_no_detection_snapshot_time = current_time

                # Cooldown check (only affects Telegram alerts, not tiered snapshots)
                if not self._check_cooldown():
                    continue

                if not person_detections:
                    continue

                # Take the highest confidence detection
                best = max(person_detections, key=lambda d: d["confidence"])
                self._last_detection_time = datetime.now()

                # Update status — always create a fresh object; never mutate after update_camera_status()
                status = CameraStatus(
                    camera_id=self.camera_id,
                    name=self.name,
                    online=True,
                    last_frame_time=status.last_frame_time,
                    fps=self._fps,
                    total_detections=status.total_detections + 1,
                    last_detection_time=self._last_detection_time,
                )
                self.shared_state.update_camera_status(status)

                # Legacy snapshot removed - use tiered snapshot path instead
                # (tiered snapshot for confirmed_45plus is saved earlier in the low_detections loop)
                snapshot_path = confirmed_snapshot_path if confirmed_snapshot_path else None

                # Record detection
                detection_obj = Detection(
                    camera_id=self.camera_id,
                    camera_name=self.name,
                    timestamp=self._last_detection_time,
                    confidence=best["confidence"],
                    bbox=best["bbox"],
                    snapshot_path=snapshot_path,
                )
                self.shared_state.add_detection(detection_obj)

                # Send Telegram alert (only for confirmed tier)
                if self.notifier:
                    self.notifier.send_alert(detection_obj)

                logger.info(f"[{self.name}] Persoană Detectată! Probabilitate: {best['confidence']:.2f}")

                time.sleep(0.1)

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}")
        finally:
            self._disconnect()
            status = CameraStatus(camera_id=self.camera_id, name=self.name, online=False)
            self.shared_state.update_camera_status(status)
            logger.info(f"[{self.name}] Camera thread stopped")
