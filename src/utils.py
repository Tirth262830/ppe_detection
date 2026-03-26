"""Utility functions for the PPE compliance detection system."""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(directory: str) -> None:
    """Create a directory if it does not already exist."""
    if directory:
        Path(directory).mkdir(parents=True, exist_ok=True)


def calculate_iou(box1: list, box2: list) -> float:
    """Calculate intersection over union."""
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
    """Check whether one bounding box overlaps enough with another."""
    return calculate_iou(inner_box, outer_box) >= iou_threshold


def draw_bbox(
    image: np.ndarray,
    bbox: list,
    label: str,
    color: tuple,
    thickness: int = 2,
    text_size: float = 0.6,
) -> np.ndarray:
    """Draw a labeled bounding box."""
    x1, y1, x2, y2 = map(int, bbox)

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        thickness,
    )

    cv2.rectangle(
        image,
        (x1, y1 - text_height - baseline - 6),
        (x1 + text_width, y1),
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


def save_violation_image(
    image: np.ndarray,
    violations_dir: str,
    violation_info: dict,
) -> str:
    """Save an annotated violation image."""
    ensure_dir(violations_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"violation_{timestamp}.jpg"
    path = os.path.join(violations_dir, filename)

    cv2.imwrite(path, image)
    return path


def log_violation(log_file: str, violation_data: dict) -> None:
    """Append a violation record to a JSON log file safely."""
    ensure_dir(os.path.dirname(log_file))

    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        logs = []
    else:
        try:
            with open(log_file, "r", encoding="utf-8") as file:
                logs = json.load(file)
        except (json.JSONDecodeError, OSError):
            logs = []

    violation_data.setdefault("timestamp", datetime.now().isoformat())
    logs.append(violation_data)

    with open(log_file, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4)


def format_confidence(confidence: float) -> str:
    """Format a confidence score."""
    return f"{confidence * 100:.1f}%"


def resize_with_aspect_ratio(
    image: np.ndarray,
    max_width: int = 1280,
    max_height: int = 720,
) -> np.ndarray:
    """Resize while preserving aspect ratio."""
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)

    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return image


def create_detection_summary(detections: dict) -> dict:
    """Create summary counts from normalized detections."""
    total_objects = sum(len(items) for items in detections.values())

    return {
        "total_detections": total_objects,
        "person": len(detections.get("person", [])),
        "helmet": len(detections.get("helmet", [])),
        "vest": len(detections.get("vest", [])),
        "without_helmet": len(detections.get("without_helmet", [])),
        "without_vest": len(detections.get("without_vest", [])),
    }
