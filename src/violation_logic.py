"""Violation logic for person-aware PPE compliance checks."""

from typing import Dict, List

import numpy as np

from src.utils import calculate_iou


class ViolationDetector:
    """Evaluate PPE compliance only when a person detection exists."""

    def __init__(self, config: dict):
        self.config = config
        violation_config = config.get("violation", {})
        self.person_ppe_iou_threshold = float(violation_config.get("person_ppe_iou_threshold", 0.05))
        self.person_center_margin = float(violation_config.get("person_center_margin", 0.05))

        print("Violation Detector initialized")

    @staticmethod
    def _bbox_center(bbox: List[float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _center_inside_person(self, person_bbox: List[float], item_bbox: List[float]) -> bool:
        px1, py1, px2, py2 = person_bbox
        cx, cy = self._bbox_center(item_bbox)
        margin_x = (px2 - px1) * self.person_center_margin
        margin_y = (py2 - py1) * self.person_center_margin
        return (px1 - margin_x) <= cx <= (px2 + margin_x) and (py1 - margin_y) <= cy <= (py2 + margin_y)

    def _is_associated(self, person_bbox: List[float], item_bbox: List[float]) -> bool:
        """Associate PPE with a person by center containment or IoU."""
        return self._center_inside_person(person_bbox, item_bbox) or (
            calculate_iou(person_bbox, item_bbox) >= self.person_ppe_iou_threshold
        )

    @staticmethod
    def _total_detections(detections: Dict[str, List[dict]]) -> int:
        return sum(len(items) for items in detections.values())

    def check_violations(self, detections: Dict[str, List[dict]]) -> Dict:
        """Check compliance when person detections are available."""
        persons = detections.get("person", [])
        helmets = detections.get("helmet", [])
        vests = detections.get("vest", [])
        total_detections = self._total_detections(detections)

        if not persons:
            result = {
                "violations": [],
                "violation_count": 0,
                "compliance_rate": 0.0,
                "person_count": 0,
                "helmet_count": len(helmets),
                "vest_count": len(vests),
                "total_detections": total_detections,
                "person_results": [],
                "status": "person_missing",
            }
            print("Violation Result:", result)
            return result

        violations: List[dict] = []
        person_results: List[dict] = []
        compliant_persons = 0

        for index, person in enumerate(persons, start=1):
            person_bbox = person["bbox"]
            has_helmet = any(self._is_associated(person_bbox, item["bbox"]) for item in helmets)
            has_vest = any(self._is_associated(person_bbox, item["bbox"]) for item in vests)

            person_violations: List[str] = []
            if not has_helmet:
                person_violations.append("without_helmet")
            if not has_vest:
                person_violations.append("without_vest")

            if not person_violations:
                compliant_persons += 1

            person_results.append(
                {
                    "person_id": index,
                    "bbox": person_bbox,
                    "has_helmet": has_helmet,
                    "has_vest": has_vest,
                    "violations": person_violations,
                    "source": person.get("source", "unknown"),
                }
            )

            for violation_type in person_violations:
                violations.append(
                    {
                        "type": violation_type,
                        "bbox": person_bbox,
                        "conf": person.get("confidence", person.get("conf", 0.0)),
                        "person_id": index,
                    }
                )

        result = {
            "violations": violations,
            "violation_count": len(violations),
            "compliance_rate": (compliant_persons / len(persons)) * 100.0,
            "person_count": len(persons),
            "helmet_count": len(helmets),
            "vest_count": len(vests),
            "total_detections": total_detections,
            "person_results": person_results,
            "status": "evaluated",
        }

        print("Violation Result:", result)
        return result

    def get_violation_summary(self, violation_result: Dict) -> str:
        """Generate a readable summary."""
        summary = f"Total Detections: {violation_result.get('total_detections', 0)}\n"
        summary += f"Persons Evaluated: {violation_result.get('person_count', 0)}\n"
        summary += f"Violations: {violation_result.get('violation_count', 0)}\n"
        summary += f"Compliance Rate: {violation_result.get('compliance_rate', 0.0):.1f}%\n"
        return summary

    def create_alert_message(self, violation_result: Dict) -> str:
        """Create a concise alert message."""
        if violation_result.get("status") == "person_missing":
            return "No person detected; compliance not evaluated"
        if violation_result["violation_count"] == 0:
            return "PPE compliant"
        if violation_result["violation_count"] == 1:
            return "1 PPE violation detected"
        return f"{violation_result['violation_count']} PPE violations detected"


def annotate_violations(
    image: np.ndarray,
    detections: Dict[str, List[dict]],
    violation_result: Dict,
    config: dict,
) -> np.ndarray:
    """Draw detections and person-level violations."""
    from src.utils import draw_bbox, format_confidence

    annotated = image.copy()
    colors = config["visualization"]["colors"]
    thickness = config["visualization"]["bbox_thickness"]
    text_size = config["visualization"]["text_size"]

    for class_name, items in detections.items():
        for det in items:
            confidence = det.get("confidence", det.get("conf", 0.0))
            label = f"{class_name} {format_confidence(confidence)}"
            annotated = draw_bbox(
                annotated,
                det["bbox"],
                label,
                tuple(colors.get(class_name, (255, 255, 255))),
                thickness,
                text_size,
            )

    for violation in violation_result.get("violations", []):
        annotated = draw_bbox(
            annotated,
            violation["bbox"],
            violation["type"].replace("_", " "),
            tuple(colors.get(violation["type"], (0, 0, 255))),
            thickness + 1,
            text_size,
        )

    return annotated
