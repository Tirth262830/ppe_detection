"""
YOLO Detector Wrapper for PPE Detection
Final version (PyTorch < 2.6 compatible)
"""

from ultralytics import YOLO
import numpy as np
from typing import Dict, List
import torch


class PPEDetector:
    """Wrapper class for YOLO-based PPE detection"""

    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.model_path = model_path
        self.conf_threshold = config["model"]["confidence_threshold"]
        self.iou_threshold = config["model"]["iou_threshold"]
        self.device = config["model"]["device"]

        print(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        # Device handling
        if self.device == "cpu":
            self.model.to("cpu")
        elif torch.cuda.is_available():
            self.model.to(self.device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")
            self.model.to("cpu")

        # Class mappings
        self.class_names = {
            config["classes"]["helmet"]: "helmet",
            config["classes"]["vest"]: "vest",
            config["classes"]["without_helmet"]: "without_helmet",
            config["classes"]["without_vest"]: "without_vest",
        }

        print("✅ PPE Detector initialized successfully")

    def detect(self, image: np.ndarray) -> Dict[str, List]:
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections = {
            "helmet": [],
            "vest": [],
            "without_helmet": [],
            "without_vest": [],
        }

        if not results:
            return detections

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            if cls_id in self.class_names:
                detections[self.class_names[cls_id]].append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": conf,
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id],
                })

        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[Dict[str, List]]:
        return [self.detect(img) for img in images]