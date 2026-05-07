"""
Main module for DVR Guard.
Wires together all components: state, cameras, detector, notifier, and dashboard.
"""

import yaml
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

from state import SharedState
from detector import PersonDetector
from camera import CameraThread, MotionConfig
from notifier import TelegramNotifier
from dashboard import create_app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = _PROJECT_ROOT / "config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_motion_config(global_motion: dict, camera_motion: dict) -> MotionConfig:
    """Merge global motion config with per-camera overrides."""
    merged = global_motion.copy()
    merged.update(camera_motion)  # Camera-specific overrides global
    return MotionConfig.from_dict(merged)


def main():
    """Main entry point."""
    logger.info("Starting DVR Guard...")

    # Load configuration
    config = load_config()

    # Initialize shared state
    shared_state = SharedState()
    shared_state.set_pipeline_start_time(datetime.now())
    logger.info("SharedState initialized")

    # Initialize person detector
    detection_config = config.get("detection", {})
    detector = PersonDetector(
        model_path=detection_config.get("model_path", str(_PROJECT_ROOT / "models/yolo26n.onnx")),
        confidence_threshold=detection_config.get("confidence_threshold", 0.45),
        nms_iou_threshold=detection_config.get("nms_iou_threshold", 0.45),
        person_class_id=detection_config.get("person_class_id", 0),
        input_size=detection_config.get("input_size", 640),
        shared_state=shared_state,
    )
    logger.info("PersonDetector initialized")

    # Initialize notifier
    telegram_config = config.get("telegram", {})
    notifier = TelegramNotifier(
        token=telegram_config.get("token"),
        chat_id=telegram_config.get("chat_id"),
        subscriber_password=telegram_config.get("subscriber_password"),
    )
    logger.info("TelegramNotifier initialized")

    # Initialize and start dashboard (before camera threads)
    dashboard_config = config.get("dashboard", {})
    app = create_app(shared_state, dashboard_config)
    dashboard_port = dashboard_config.get("port", 8080)
    dashboard_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=dashboard_port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
        name="Dashboard",
    )
    dashboard_thread.start()
    logger.info(f"Dashboard started on port {dashboard_port}")

    # Global settings
    global_motion = config.get("motion", {})
    cooldown_seconds = config.get("cooldown_seconds", 60)
    snapshot_dir = config.get("snapshot_dir", str(_PROJECT_ROOT / "snapshots"))
    max_snapshots_per_folder = config.get("max_snapshots_per_folder", 500)
    debug_config = config.get("debug", {})
    tiered_snapshots = debug_config.get("tiered_snapshots", False)

    # Stop event for coordinated shutdown
    stop_event = threading.Event()

    # Create and start camera threads
    camera_threads = []
    for cam_config in config.get("cameras", []):
        cam_id = cam_config["id"]
        cam_name = cam_config["name"]
        rtsp_url = cam_config["rtsp"]  # Sub-stream URLs from config.yaml

        # Merge motion config (global + per-camera overrides)
        cam_motion = cam_config.get("motion", {})
        motion_config = merge_motion_config(global_motion, cam_motion)

        # Per-camera confidence threshold override
        confidence_threshold = cam_config.get(
            "confidence_threshold",
            detection_config.get("confidence_threshold", 0.45),
        )

        # Per-camera cooldown override (with global fallback)
        cooldown = cam_config.get("cooldown_seconds", cooldown_seconds)

        # Create camera thread
        cam_thread = CameraThread(
            camera_id=cam_id,
            name=cam_name,
            rtsp_url=rtsp_url,
            shared_state=shared_state,
            motion_config=motion_config,
            detector=detector,
            notifier=notifier,
            confidence_threshold=confidence_threshold,
            cooldown_seconds=cooldown,
            snapshot_dir=snapshot_dir,
            tiered_snapshots=tiered_snapshots,
            max_snapshots_per_folder=max_snapshots_per_folder,
            stop_event=stop_event,
        )

        cam_thread.start()
        camera_threads.append(cam_thread)
        logger.info(f"Started camera thread: {cam_name} (ID: {cam_id}, cooldown={cooldown}s)")

    logger.info(f"DVR Guard is running. Dashboard: http://localhost:{dashboard_port}")
    logger.info("Press Ctrl+C to stop.")

    try:
        # Keep main thread alive
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown signal received...")
        stop_event.set()

    # Cleanup
    logger.info("Stopping camera threads...")
    for t in camera_threads:
        t.join(timeout=10.0)

    logger.info("Stopping notifier...")
    notifier.stop()

    logger.info("Closing detector...")
    detector.close()

    logger.info("DVR Guard stopped.")


if __name__ == "__main__":
    main()
