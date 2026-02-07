"""
Frame Processor – PPE Detection System
FINAL STABLE VERSION (Object-based PPE)
"""

import cv2
import numpy as np
from typing import Dict, Optional
from datetime import datetime

from src.detector import PPEDetector
from src.violation_logic import ViolationDetector, annotate_violations
from src.utils import (
    save_violation_image,
    resize_with_aspect_ratio,
    ensure_dir,
)


class FrameProcessor:
    """
    Main processing pipeline for PPE Detection
    """

    def __init__(self, config: dict, model_path: str):
        self.config = config

        print("Initializing PPE Detector...")
        self.detector = PPEDetector(model_path, config)

        print("Initializing Violation Detector...")
        self.violation_detector = ViolationDetector(config)

        # Storage
        self.violations_dir = config["storage"]["violations_dir"]
        self.save_violations = config["storage"]["save_violations"]

        ensure_dir(self.violations_dir)

        print("✅ Frame Processor initialized")

    # --------------------------------------------------
    # CORE FRAME PROCESSING
    # --------------------------------------------------

    def process_frame(
        self,
        frame: np.ndarray,
        save_violation: bool = True,
        source_name: str = "unknown",
    ) -> Dict:
        """
        Process a single frame
        """

        # 1. Detection
        detections = self.detector.detect(frame)

        # 2. Violation logic
        violation_result = self.violation_detector.check_violations(detections)

        # 3. Annotation
        annotated_frame = annotate_violations(
            frame, detections, violation_result, self.config
        )

        # 4. Save violation image
        if (
            save_violation
            and self.save_violations
            and violation_result["violation_count"] > 0
        ):
            self._save_violation_image(
                annotated_frame, violation_result, source_name
            )

        return {
            "annotated_frame": annotated_frame,
            "detections": detections,
            "violations": violation_result,
            "summary": {
                "violations": violation_result["violation_count"],
                "compliance_rate": violation_result["compliance_rate"],
            },
        }

    # --------------------------------------------------
    # IMAGE PROCESSING
    # --------------------------------------------------

    def process_image(
        self, image_path: str, output_path: Optional[str] = None
    ) -> Dict:
        """
        Process a single image
        """

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        result = self.process_frame(frame, True, image_path)

        if output_path:
            cv2.imwrite(output_path, result["annotated_frame"])

        return result

    # --------------------------------------------------
    # VIDEO PROCESSING
    # --------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        display: bool = False,
        skip_frames: int = 0,
    ) -> Dict:
        """
        Process a video file
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

        stats = {
            "frames_processed": 0,
            "total_violations": 0,
            "frames_with_violations": 0,
        }

        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_id % (skip_frames + 1) != 0:
                frame_id += 1
                continue

            result = self.process_frame(
                frame, True, f"{video_path}_frame_{frame_id}"
            )

            stats["frames_processed"] += 1
            stats["total_violations"] += result["violations"]["violation_count"]

            if result["violations"]["violation_count"] > 0:
                stats["frames_with_violations"] += 1

            if writer:
                writer.write(result["annotated_frame"])

            if display:
                cv2.imshow(
                    "PPE Detection",
                    resize_with_aspect_ratio(result["annotated_frame"]),
                )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_id += 1

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        return stats

    # --------------------------------------------------
    # WEBCAM PROCESSING
    # --------------------------------------------------

    def process_webcam(self, camera_id: int = 0):
        """
        Real-time webcam processing
        """
        print(f"Starting webcam (camera {camera_id})")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(
                frame,
                save_violation=True,
                source_name=f"webcam_{camera_id}"
            )

            display_frame = resize_with_aspect_ratio(
                result["annotated_frame"]
            )

            cv2.imshow("PPE Detection - Webcam", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Webcam stopped")

    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------

    def _save_violation_image(
        self,
        frame: np.ndarray,
        violation_result: Dict,
        source_name: str,
    ) -> None:
        """
        Save violation image only (stable version)
        """

        violation_info = {
            "source": source_name,
            "violation_count": violation_result["violation_count"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

        save_violation_image(frame, self.violations_dir, violation_info)