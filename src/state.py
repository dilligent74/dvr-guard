"""
Shared state module for DVR Guard.
Thread-safe state management for camera status and recent detections.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CameraStatus:
    """Status information for a single camera."""
    camera_id: int
    name: str
    online: bool = False
    last_frame_time: Optional[datetime] = None
    last_motion_time: Optional[datetime] = None
    last_detection_time: Optional[datetime] = None
    total_detections: int = 0
    fps: float = 0.0


@dataclass
class Detection:
    """A single person detection event."""
    camera_id: int
    camera_name: str
    timestamp: datetime
    confidence: float
    bbox: tuple  # (x, y, w, h)
    snapshot_path: Optional[str] = None


class SharedState:
    """
    Thread-safe shared state for DVR Guard.
    Holds camera status and recent detections accessible by all threads.
    """

    def __init__(self, max_recent_detections: int = 100):
        self._lock = threading.Lock()
        self._camera_statuses: dict[int, CameraStatus] = {}
        self._recent_detections: list[Detection] = []
        self._max_recent_detections = max_recent_detections
        # New fields for system health monitoring
        self._stream_status: dict[str, datetime] = {}  # key=camera_name, value=last successful frame timestamp
        self._yolo_last_inference: Optional[datetime] = None
        self._pipeline_start_time: Optional[datetime] = None

    def update_camera_status(self, status: CameraStatus) -> None:
        """Update or add a camera's status."""
        with self._lock:
            self._camera_statuses[status.camera_id] = status

    def get_camera_status(self, camera_id: int) -> Optional[CameraStatus]:
        """Get status for a specific camera."""
        with self._lock:
            return self._camera_statuses.get(camera_id)

    def get_all_camera_statuses(self) -> dict[int, CameraStatus]:
        """Get all camera statuses."""
        with self._lock:
            return dict(self._camera_statuses)

    def add_detection(self, detection: Detection) -> None:
        """Add a new detection to recent detections list."""
        with self._lock:
            self._recent_detections.append(detection)
            # Trim to max size
            if len(self._recent_detections) > self._max_recent_detections:
                self._recent_detections = self._recent_detections[-self._max_recent_detections:]

    def get_recent_detections(self, limit: Optional[int] = None) -> list[Detection]:
        """Get recent detections, optionally limited to last N."""
        with self._lock:
            detections = list(self._recent_detections)
        if limit is not None:
            return detections[-limit:]
        return detections

    def clear_detections(self) -> None:
        """Clear all recent detections."""
        with self._lock:
            self._recent_detections.clear()

    def get_summary(self) -> dict:
        """Get a summary of current state for dashboard."""
        with self._lock:
            return {
                "cameras": {
                    cam_id: {
                        "name": status.name,
                        "online": status.online,
                        "fps": status.fps,
                        "total_detections": status.total_detections,
                        "last_detection": status.last_detection_time.isoformat() if status.last_detection_time else None,
                    }
                    for cam_id, status in self._camera_statuses.items()
                },
                "recent_detections_count": len(self._recent_detections),
            }

    # --- New methods for system health monitoring ---

    def touch_stream(self, camera_name: str) -> None:
        """Update stream_status with current timestamp for a camera."""
        with self._lock:
            self._stream_status[camera_name] = datetime.now()

    def get_stream_status(self) -> dict[str, datetime]:
        """Get copy of stream status dict."""
        with self._lock:
            return dict(self._stream_status)

    def touch_yolo_inference(self) -> None:
        """Update yolo_last_inference to current time."""
        with self._lock:
            self._yolo_last_inference = datetime.now()

    def get_yolo_last_inference(self) -> Optional[datetime]:
        """Get last YOLO inference timestamp."""
        with self._lock:
            return self._yolo_last_inference

    def set_pipeline_start_time(self, dt: datetime) -> None:
        """Set pipeline start time only if not already set."""
        with self._lock:
            if self._pipeline_start_time is None:
                self._pipeline_start_time = dt

    def get_pipeline_start_time(self) -> Optional[datetime]:
        """Get pipeline start time."""
        with self._lock:
            return self._pipeline_start_time
