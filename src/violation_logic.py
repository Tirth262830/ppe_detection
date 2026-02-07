"""
PPE Compliance Violation Logic (Object-Based)
Compatible with datasets WITHOUT 'person' class
"""

from typing import Dict, List
import numpy as np


class ViolationDetector:
    """Detects PPE compliance violations based on detected violation classes"""

    def __init__(self, config: dict):
        """
        Initialize Violation Detector

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.violation_classes = config["violation"]["violation_classes"]

        print("Violation Detector initialized")
        print(f"Violation classes: {', '.join(self.violation_classes)}")

    def check_violations(self, detections: Dict[str, List]) -> Dict:
        """
        Check for PPE violations

        Logic:
        - Any detection of `without_*` is a violation
        - Compliance is absence of violations

        Args:
            detections: Output from detector.detect()

        Returns:
            Dictionary with violation information
        """

        violations = []

        for cls in self.violation_classes:
            for det in detections.get(cls, []):
                violations.append({
                    "type": cls,
                    "bbox": det["bbox"],
                    "conf": det["conf"],
                })

        violation_count = len(violations)
        compliance_rate = 100.0 if violation_count == 0 else 0.0

        return {
            "violations": violations,
            "violation_count": violation_count,
            "compliance_rate": compliance_rate,
        }

    def get_violation_summary(self, violation_result: Dict) -> str:
        """Generate human-readable violation summary"""

        violations = violation_result["violation_count"]
        compliance = violation_result["compliance_rate"]

        summary = f"Violations: {violations}\n"
        summary += f"Compliance Rate: {compliance:.1f}%\n"

        if violations > 0:
            summary += "\nViolation Details:\n"
            for i, viol in enumerate(violation_result["violations"], 1):
                summary += (
                    f"  {i}. {viol['type']} "
                    f"(conf: {viol['conf']:.2f})\n"
                )

        return summary

    def create_alert_message(self, violation_result: Dict) -> str:
        """Create concise alert message for dashboard"""

        violations = violation_result["violation_count"]

        if violations == 0:
            return "✅ PPE compliant"
        elif violations == 1:
            return "⚠️ 1 PPE violation detected"
        else:
            return f"🚨 {violations} PPE violations detected"


def annotate_violations(
    image: np.ndarray,
    detections: Dict,
    violation_result: Dict,
    config: dict,
) -> np.ndarray:
    """
    Draw bounding boxes and violation labels on image
    """

    from src.utils import draw_bbox, format_confidence

    annotated = image.copy()
    colors = config["visualization"]["colors"]
    thickness = config["visualization"]["bbox_thickness"]
    text_size = config["visualization"]["text_size"]

    # Draw all detections
    for cls, items in detections.items():
        for det in items:
            label = f"{cls} {format_confidence(det['conf'])}"
            annotated = draw_bbox(
                annotated,
                det["bbox"],
                label,
                tuple(colors.get(cls, (255, 255, 255))),
                thickness,
                text_size,
            )

    return annotated