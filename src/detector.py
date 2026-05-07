"""
Person detector module for DVR Guard.
YOLO26n ONNX wrapper -- exported WITH nms=True, output shape (1, 300, 6).
"""

import cv2
import numpy as np
import logging
import threading
import argparse
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PersonDetector:
    """
    YOLO26n ONNX wrapper for person detection.
    Single inference thread -- do NOT parallelize YOLO calls.

    Expected ONNX output: (1, 300, 6)
      300 = max detections (padded with zeros)
      6   = [x1, y1, x2, y2, confidence, class_id]
    NMS is already applied inside the model -- no manual NMS needed.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.45,
        nms_iou_threshold: float = 0.45,   # INERT - kept for config compatibility only
                                           # Model exported with nms=True, so NMS is already applied
        person_class_id: int = 0,
        input_size: int = 640,
        shared_state=None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.person_class_id = person_class_id
        self.input_size = input_size
        self._shared_state = shared_state
        self._session = None
        self._first_inference = True
        self._inference_lock = threading.Lock()
        self._load_model()

    def _load_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            )

        logger.info(f"Loading ONNX model from {self.model_path}")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self._session = ort.InferenceSession(self.model_path)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        input_shape = self._session.get_inputs()[0].shape
        logger.info(f"Model input shape: {input_shape}")
        logger.info(f"Model output names: {self._output_names}")

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """
        Letterbox-resize frame to input_size x input_size.
        Returns: (blob, scale, (pad_x, pad_y))
        """
        h, w = frame.shape[:2]

        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)       # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # add batch dim

        return blob, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        outputs: list,
        scale: float,
        pad_offset: tuple,
        orig_shape: tuple,
        confidence_threshold: float,
    ) -> list[dict]:
        """
        Postprocess YOLO26n output for nms=True export.

        Output shape: (1, 300, 6)
          - 300 rows, most zero-padded (confidence == 0)
          - 6 columns: [x1, y1, x2, y2, confidence, class_id]
          - Coordinates are in letterboxed input pixel space (0..input_size)

        Steps:
          1. Filter by confidence threshold and class_id
          2. Unpad: subtract letterbox offsets
          3. Rescale: divide by scale factor to get original image coords
          4. Clamp to image bounds and discard degenerate boxes
        """
        orig_h, orig_w = orig_shape
        pad_x, pad_y = pad_offset

        detections = outputs[0][0]   # shape: (300, 6)

        results = []
        for det in detections:
            x1, y1, x2, y2, confidence, class_id = det

            # Zero-padded rows have confidence == 0; skip low-confidence hits
            if confidence < confidence_threshold:
                continue

            # Only keep persons (class_id 0 in COCO)
            if int(round(class_id)) != self.person_class_id:
                continue

            # Unpad + rescale: letterboxed input space -> original image space
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clamp to original image bounds
            x1 = max(0.0, min(float(x1), orig_w))
            y1 = max(0.0, min(float(y1), orig_h))
            x2 = max(0.0, min(float(x2), orig_w))
            y2 = max(0.0, min(float(y2), orig_h))

            w = x2 - x1
            h = y2 - y1

            # Discard degenerate boxes (clipped to a point or line)
            if w < 2 or h < 2:
                continue

            results.append({
                "bbox":       (int(x1), int(y1), int(w), int(h)),
                "confidence": float(confidence),
                "class_id":   int(round(class_id)),
            })

        return results

    def infer(
        self,
        frame: np.ndarray,
    ) -> tuple:
        """
        Run inference only (preprocess + session.run).
        Returns (outputs, scale, pad_offset, orig_shape) for later postprocessing.
        Use this to avoid re-running inference when trying multiple confidence thresholds.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded.")

        blob, scale, pad_offset = self._preprocess(frame)
        orig_shape = frame.shape[:2]

        with self._inference_lock:
            outputs = self._session.run(self._output_names, {self._input_name: blob})

        if self._first_inference:
            shape = outputs[0].shape
            logger.info(f"First inference output shape: {shape}")
            if len(shape) != 3 or shape[2] != 6:
                logger.warning(
                    f"Unexpected output shape {shape}. "
                    f"Expected (1, N, 6) for nms=True export."
                )
            self._first_inference = False

        # Update YOLO last inference timestamp
        if self._shared_state is not None:
            self._shared_state.touch_yolo_inference()

        return outputs, scale, pad_offset, orig_shape

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> list[dict]:
        """
        Run person detection on a frame.
        Returns list of dicts: {bbox: (x,y,w,h), confidence: float, class_id: int}

        Args:
            frame: BGR image, any resolution.
            confidence_threshold: Per-call override; falls back to instance default.
        """
        if self._session is None:
            raise RuntimeError("Model not loaded.")

        # Resolve threshold once, pass it down -- never mutate instance state
        threshold = confidence_threshold if confidence_threshold is not None \
                    else self.confidence_threshold

        outputs, scale, pad_offset, orig_shape = self.infer(frame)
        return self._postprocess(outputs, scale, pad_offset, orig_shape, threshold)

    def close(self):
        """Release ONNX session resources."""
        if self._session is not None:
            del self._session
            self._session = None   # prevent accidental reuse after close()


# ---------------------------------------------------------------------------
# CLI test mode
# ---------------------------------------------------------------------------

def test_image(image_path: str, model_path: str, confidence: float = 0.45):
    """Run detector on a single image and save annotated output."""
    logging.basicConfig(level=logging.INFO)

    detector = PersonDetector(
        model_path=model_path,
        confidence_threshold=confidence,
    )

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Failed to load image: {image_path}")
        return

    print(f"Image shape: {frame.shape}")
    detections = detector.detect(frame)
    print(f"Detections: {len(detections)}")

    for det in detections:
        x, y, w, h = det["bbox"]
        conf = det["confidence"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Person {conf:.2f}",   # will now always be 0.00-1.00
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        print(f"  Person at ({x}, {y}, {w}, {h})  conf={conf:.2f}")

    output_path = "test_output.jpg"
    cv2.imwrite(output_path, frame)
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test YOLO26n person detector")
    parser.add_argument("--image",      required=True,                 help="Path to test image")
    parser.add_argument("--model",      default="models/yolo26n.onnx", help="Path to ONNX model")
    parser.add_argument("--confidence", type=float, default=0.45,      help="Confidence threshold")
    args = parser.parse_args()

    test_image(args.image, args.model, args.confidence)


