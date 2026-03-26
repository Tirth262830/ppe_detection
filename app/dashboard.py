"""
Streamlit dashboard for the PPE compliance detection project.

The dashboard is aligned with the current object-based pipeline:
- detections are class-based (`helmet`, `vest`, `without_helmet`, `without_vest`)
- violations are direct `without_*` detections
- violation images may be saved to disk
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add project root to import path
sys.path.append(str(Path(__file__).parent.parent))

from src.processor import FrameProcessor
from src.utils import create_detection_summary, load_config, log_violation


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def get_dashboard_config(config: dict) -> dict:
    """Return dashboard config with safe fallbacks."""
    return config.get(
        "dashboard",
        {
            "title": "PPE Compliance Detection System",
            "page_icon": "PPE",
            "layout": "wide",
        },
    )


def get_config_version() -> int:
    """Use config file mtime to invalidate Streamlit cache when config changes."""
    return CONFIG_PATH.stat().st_mtime_ns


BASE_CONFIG = load_config(str(CONFIG_PATH))
PAGE_CONFIG = get_dashboard_config(BASE_CONFIG)

st.set_page_config(
    page_title=PAGE_CONFIG.get("title", "PPE Compliance Detection System"),
    page_icon=PAGE_CONFIG.get("page_icon", "PPE"),
    layout=PAGE_CONFIG.get("layout", "wide"),
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --bg-soft: #f5efe6;
            --panel: #fffaf3;
            --ink: #1f2937;
            --accent: #b45309;
            --accent-dark: #7c2d12;
            --alert: #b91c1c;
            --ok: #166534;
            --line: #eadfce;
        }
        .main-header {
            font-size: 2.2rem;
            text-align: center;
            padding: 1rem 1.25rem;
            background: linear-gradient(135deg, #1f2937 0%, #9a3412 100%);
            color: white;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            letter-spacing: 0.02em;
        }
        .status-card {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: var(--panel);
            color: var(--ink);
        }
        .violation-alert {
            background-color: #fef2f2;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #fecaca;
            color: var(--alert);
            font-weight: 600;
        }
        .compliant-alert {
            background-color: #f0fdf4;
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #bbf7d0;
            color: var(--ok);
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_image(image: object, caption: str | None = None) -> None:
    """Render images across Streamlit versions."""
    try:
        st.image(image, caption=caption, use_container_width=True)
    except TypeError:
        # Streamlit 1.28 uses `use_column_width` for responsive images.
        st.image(image, caption=caption, use_column_width=True)


@st.cache_resource
def load_processor(_config_version: int) -> tuple[FrameProcessor, dict]:
    """Load config and processor once per session."""
    config = load_config(str(CONFIG_PATH))
    model_path = config["model"]["weights_path"]

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.info("Train or place the model first, then rerun the dashboard.")
        st.stop()

    processor = FrameProcessor(config, model_path)
    return processor, config


def apply_runtime_settings(
    processor: FrameProcessor,
    config: dict,
    confidence_threshold: float,
    iou_threshold: float,
    save_violations: bool,
) -> None:
    """Apply sidebar settings to the cached processor instance."""
    config["model"]["confidence_threshold"] = confidence_threshold
    config["model"]["iou_threshold"] = iou_threshold
    config["storage"]["save_violations"] = save_violations

    processor.detector.conf_threshold = confidence_threshold
    processor.detector.iou_threshold = iou_threshold
    processor.save_violations = save_violations


def build_image_log_entry(result: dict, source_name: str, saved_path: str | None) -> dict:
    """Build a stable log record from an image inference result."""
    detection_summary = create_detection_summary(result["detections"])
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source_name,
        "source_type": "image",
        "violation_count": result["violations"]["violation_count"],
        "compliance_rate": result["violations"]["compliance_rate"],
        "image_path": saved_path,
        "detection_summary": detection_summary,
        "violations": result["violations"]["violations"],
    }


def append_image_log(processor: FrameProcessor, result: dict, source_name: str) -> None:
    """Persist image detection metadata for the statistics page."""
    log_file = processor.config["storage"].get("log_file")
    if not log_file:
        return

    saved_path = None
    if result["violations"]["violation_count"] > 0:
        violations_dir = Path(processor.violations_dir)
        if violations_dir.exists():
            images = sorted(
                violations_dir.glob("*.jpg"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if images:
                saved_path = str(images[0])

    log_violation(log_file, build_image_log_entry(result, source_name, saved_path))


def load_logs(log_file: str) -> list[dict]:
    """Load the violation log file safely."""
    if not log_file or not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        return []

    try:
        with open(log_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def summarize_logs(logs: list[dict]) -> dict:
    """Aggregate statistics from saved logs."""
    total_records = len(logs)
    total_violations = sum(int(log.get("violation_count", 0)) for log in logs)
    records_with_violations = sum(1 for log in logs if int(log.get("violation_count", 0)) > 0)
    avg_violations = total_violations / total_records if total_records else 0.0
    return {
        "total_records": total_records,
        "total_violations": total_violations,
        "records_with_violations": records_with_violations,
        "average_violations_per_record": avg_violations,
    }


def list_violation_images(violations_dir: str) -> list[Path]:
    """Return saved violation images sorted by most recent first."""
    directory = Path(violations_dir)
    if not directory.exists():
        return []

    image_files = [
        path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    return sorted(image_files, key=lambda path: path.stat().st_mtime, reverse=True)


def render_detection_details(result: dict, show_confidence: bool) -> None:
    """Render detection and violation details for image results."""
    detection_summary = create_detection_summary(result["detections"])
    violations = result["violations"]["violation_count"]
    compliance = result["violations"]["compliance_rate"]
    total_objects = result.get("summary", {}).get(
        "total_detections",
        detection_summary["total_detections"],
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Objects Detected", total_objects)
    with metric_col2:
        st.metric("Violations", violations)
    with metric_col3:
        st.metric("Compliance Rate", f"{compliance:.1f}%")

    if total_objects == 0:
        st.warning("No objects were confidently detected in this frame.")
    elif violations > 0:
        st.markdown(
            f'<div class="violation-alert">{violations} PPE violation(s) detected.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="compliant-alert">No PPE violations detected in this frame.</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Detection Summary", expanded=True):
        st.json(detection_summary)

    if result["violations"]["violations"]:
        with st.expander("Violation Details", expanded=True):
            for index, violation in enumerate(result["violations"]["violations"], start=1):
                label = violation["type"].replace("_", " ").title()
                confidence = violation.get("conf", 0.0)
                if show_confidence:
                    st.write(f"{index}. {label} ({confidence:.2%})")
                else:
                    st.write(f"{index}. {label}")


def main() -> None:
    """Main dashboard application."""
    st.markdown(
        '<div class="main-header">PPE Compliance Detection System</div>',
        unsafe_allow_html=True,
    )

    try:
        processor, config = load_processor(get_config_version())
    except Exception as exc:
        st.error(f"Failed to load the system: {exc}")
        st.stop()

    with st.sidebar:
        st.title("Control Panel")

        mode = st.selectbox(
            "Select Mode",
            ["Image Upload", "Video Upload", "Webcam Guide", "Statistics"],
        )

        st.markdown("---")
        st.subheader("Runtime Settings")

        save_violations = st.checkbox(
            "Save violation images",
            value=config["storage"].get("save_violations", True),
        )
        show_confidence = st.checkbox("Show confidence values", value=True)

        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=1.00,
            value=float(config["model"]["confidence_threshold"]),
            step=0.05,
        )

        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.10,
            max_value=0.90,
            value=float(config["model"]["iou_threshold"]),
            step=0.05,
        )

        apply_runtime_settings(
            processor,
            config,
            confidence_threshold,
            iou_threshold,
            save_violations,
        )

        st.markdown("---")
        st.subheader("System Info")
        st.markdown(
            f"""
            <div class="status-card">
                <div><strong>Model:</strong> {Path(processor.detector.model_path).name}</div>
                <div><strong>Person Model:</strong> {Path(processor.detector.person_model_path).name}</div>
                <div><strong>Device:</strong> {processor.detector.device}</div>
                <div><strong>Confidence:</strong> {processor.detector.conf_threshold:.2f}</div>
                <div><strong>Classes:</strong> person, {", ".join(sorted(set(processor.detector.ppe_class_names.values())))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if mode == "Image Upload":
        image_upload_mode(processor, show_confidence)
    elif mode == "Video Upload":
        video_upload_mode(processor)
    elif mode == "Webcam Guide":
        webcam_mode()
    else:
        statistics_mode(processor)


def image_upload_mode(processor: FrameProcessor, show_confidence: bool) -> None:
    """Image upload and processing mode."""
    st.header("Image Analysis")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to detect PPE compliance.",
            key="image_uploader",
        )

        if uploaded_file is not None:
            current_upload_token = f"{uploaded_file.name}:{uploaded_file.size}"
            previous_upload_token = st.session_state.get("image_upload_token")
            if previous_upload_token != current_upload_token:
                st.session_state.pop("image_result", None)
                st.session_state["image_upload_token"] = current_upload_token

            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            render_image(image, caption="Original Image")

            if st.button("Run Detection", type="primary", key="detect_image"):
                with st.spinner("Processing image..."):
                    result = processor.process_frame(
                        image_bgr,
                        save_violation=processor.save_violations,
                        source_name=uploaded_file.name,
                    )
                    append_image_log(processor, result, uploaded_file.name)
                    st.session_state["image_result"] = result

    with col2:
        st.subheader("Detection Results")
        if st.button("Clear Previous Result", key="clear_image_result"):
            st.session_state.pop("image_result", None)
            st.session_state.pop("image_upload_token", None)
            st.rerun()

        result = st.session_state.get("image_result")
        if not result:
            st.info("Upload an image and run detection to see results here.")
            return

        annotated_rgb = cv2.cvtColor(result["annotated_frame"], cv2.COLOR_BGR2RGB)
        render_image(annotated_rgb, caption="Annotated Output")
        render_detection_details(result, show_confidence)


def video_upload_mode(processor: FrameProcessor) -> None:
    """Video upload and processing mode."""
    st.header("Video Analysis")

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video to process for PPE compliance.",
        key="video_uploader",
    )

    if uploaded_file is None:
        st.info("Upload a video to start processing.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.video(video_path)

    col1, col2 = st.columns(2)
    with col1:
        skip_frames = st.number_input(
            "Process every N frames",
            min_value=0,
            max_value=30,
            value=0,
            help="0 means every frame. 5 means every 6th frame is processed.",
        )
    with col2:
        save_output = st.checkbox("Save annotated video", value=False)

    if st.button("Process Video", type="primary", key="process_video"):
        output_path = None
        if save_output:
            output_path = str(
                PROJECT_ROOT / f"output_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )

        try:
            with st.spinner("Processing video. This can take a while."):
                stats = processor.process_video(
                    video_path,
                    output_path=output_path,
                    display=False,
                    skip_frames=int(skip_frames),
                )
        except Exception as exc:
            st.error(f"Video processing failed: {exc}")
        else:
            st.success("Video processing completed.")

            frames_processed = stats["frames_processed"]
            total_violations = stats["total_violations"]
            frames_with_violations = stats["frames_with_violations"]
            avg_violations = total_violations / frames_processed if frames_processed else 0.0

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Frames Processed", frames_processed)
            with metric_col2:
                st.metric("Total Violations", total_violations)
            with metric_col3:
                st.metric("Frames With Violations", frames_with_violations)
            with metric_col4:
                st.metric("Avg Violations / Frame", f"{avg_violations:.2f}")

            if output_path and os.path.exists(output_path):
                st.success(f"Annotated video saved to: {output_path}")
                with open(output_path, "rb") as file:
                    st.download_button(
                        "Download Annotated Video",
                        data=file,
                        file_name=Path(output_path).name,
                        mime="video/mp4",
                    )
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)


def webcam_mode() -> None:
    """Show CLI instructions for webcam mode."""
    st.header("Webcam Mode")
    st.info(
        """
Webcam detection runs best through the CLI because Streamlit is not ideal for
real-time local camera rendering in this project.
        """
    )

    st.code(
        "python -m src.inference --source webcam --camera 0",
        language="bash",
    )
    st.write("Press `q` in the OpenCV window to stop the webcam stream.")


def statistics_mode(processor: FrameProcessor) -> None:
    """Display saved logs and recent violation images."""
    st.header("System Statistics")

    log_file = processor.config["storage"].get("log_file", "")
    logs = load_logs(log_file)
    summary = summarize_logs(logs)
    images = list_violation_images(processor.violations_dir)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Logged Runs", summary["total_records"])
    with metric_col2:
        st.metric("Total Violations", summary["total_violations"])
    with metric_col3:
        st.metric("Runs With Violations", summary["records_with_violations"])
    with metric_col4:
        st.metric("Saved Images", len(images))

    if not logs and not images:
        st.info("No saved logs or violation images found yet.")
        return

    if logs:
        st.markdown("---")
        st.subheader("Recent Logs")

        rows = []
        for log in reversed(logs[-50:]):
            detection_summary = log.get("detection_summary", {})
            rows.append(
                {
                    "Timestamp": log.get("timestamp", ""),
                    "Source": log.get("source", ""),
                    "Type": log.get("source_type", ""),
                    "Violations": log.get("violation_count", 0),
                    "Compliance %": f'{float(log.get("compliance_rate", 0.0)):.1f}%',
                    "Person": detection_summary.get("person", 0),
                    "Helmet": detection_summary.get("helmet", 0),
                    "Vest": detection_summary.get("vest", 0),
                    "Without Helmet": detection_summary.get("without_helmet", 0),
                    "Without Vest": detection_summary.get("without_vest", 0),
                    "Image": log.get("image_path", ""),
                }
            )

        dataframe = pd.DataFrame(rows)
        st.dataframe(dataframe, use_container_width=True)
        st.download_button(
            "Download Logs JSON",
            data=json.dumps(logs, indent=2),
            file_name=f"violation_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

    if images:
        st.markdown("---")
        st.subheader("Recent Violation Images")

        columns = st.columns(3)
        for index, image_path in enumerate(images[:12]):
            with columns[index % 3]:
                image = Image.open(image_path)
                render_image(image, caption=image_path.name)


if __name__ == "__main__":
    main()
