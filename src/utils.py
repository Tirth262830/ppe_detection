"""
Utility functions for PPE Compliance Detection System
FINAL – robust, Windows-safe, JSON-safe
"""

import os
import json
import yaml
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np


# ---------------- CONFIG ---------------- #

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)


# ---------------- GEOMETRY ---------------- #

def calculate_iou(box1: list, box2: list) -> float:
    """Calculate Intersection over Union (IoU)"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def is_bbox_inside(inner_box: list, outer_box: list, iou_threshold: float = 0.3) -> bool:
    """Check if PPE bbox overlaps person bbox"""
    return calculate_iou(inner_box, outer_box) >= iou_threshold


# ---------------- DRAWING ---------------- #

def draw_bbox(
    image: np.ndarray,
    bbox: list,
    label: str,
    color: tuple,
    thickness: int = 2,
    text_size: float = 0.6,
) -> np.ndarray:
    """Draw bounding box with label"""
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, text_size, thickness
    )

    cv2.rectangle(
        image,
        (x1, y1 - th - baseline - 6),
        (x1 + tw, y1),
        color,
        -1,
    )

    cv2.putText(
        image,
        label,
        (x1, y1 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        (255, 255, 255),
        thickness,
    )

    return image


# ---------------- FILE OUTPUT ---------------- #

def save_violation_image(
    image: np.ndarray, violations_dir: str, violation_info: dict
) -> str:
    """Save violation image"""
    ensure_dir(violations_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"violation_{timestamp}.jpg"
    path = os.path.join(violations_dir, filename)

    cv2.imwrite(path, image)
    return path


def log_violation(log_file: str, violation_data: dict) -> None:
    """
    SAFE JSON logger
    - creates file if missing
    - handles empty / corrupt JSON
    """

    ensure_dir(os.path.dirname(log_file))

    # Load existing data safely
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        logs = []
    else:
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, OSError):
            logs = []

    # Add timestamp if missing
    violation_data.setdefault("timestamp", datetime.now().isoformat())

    logs.append(violation_data)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4)


# ---------------- HELPERS ---------------- #

def format_confidence(confidence: float) -> str:
    """Format confidence score"""
    return f"{confidence * 100:.1f}%"


def resize_with_aspect_ratio(
    image: np.ndarray, max_width: int = 1280, max_height: int = 720
) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)

    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


# ---------------- SUMMARY ---------------- #

def create_detection_summary(detections: dict) -> dict:
    """
    Create detection summary
    (aligned with current dataset – no 'person' class)
    """
    total_objects = sum(len(v) for v in detections.values())

    return {
        "total_detections": total_objects,
        "helmet": len(detections.get("helmet", [])),
        "vest": len(detections.get("vest", [])),
        "without_helmet": len(detections.get("without_helmet", [])),
        "without_vest": len(detections.get("without_vest", [])),
    }