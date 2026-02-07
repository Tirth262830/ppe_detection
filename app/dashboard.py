"""
PPE Compliance Detection - Streamlit Dashboard
Real-time monitoring and analysis interface
"""

import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import sys
import os
from datetime import datetime
import json
import pandas as pd
from PIL import Image
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.processor import FrameProcessor
from src.utils import load_config, ensure_dir


# Page configuration
st.set_page_config(
    page_title="PPE Compliance Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF4B4B;
    }
    .violation-alert {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #ff0000;
        color: #cc0000;
        font-weight: bold;
    }
    .compliant-alert {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #00cc00;
        color: #008000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_processor():
    """Load frame processor (cached)"""
    config = load_config()
    model_path = config['model']['weights_path']
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
        st.info("Please train the model first using: python src/train.py")
        st.stop()
    
    processor = FrameProcessor(config, model_path)
    return processor, config


def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🦺 PPE Compliance Detection System</div>', 
                unsafe_allow_html=True)
    
    # Load processor
    try:
        processor, config = load_processor()
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/security-configuration.png", 
                 width=100)
        st.title("Control Panel")
        
        # Mode selection
        mode = st.selectbox(
            "Select Mode",
            ["📸 Image Upload", "🎥 Video Upload", "📹 Webcam (Local)", "📊 Statistics"]
        )
        
        st.markdown("---")
        
        # Settings
        st.subheader("Settings")
        save_violations = st.checkbox("Save Violations", value=True)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        
        # Detection thresholds
        st.subheader("Detection Thresholds")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=config['model']['confidence_threshold'],
            step=0.05
        )
        
        iou_threshold = st.slider(
            "IoU Threshold (PPE Association)",
            min_value=0.1,
            max_value=0.9,
            value=config['violation']['iou_threshold'],
            step=0.05
        )
        
        st.markdown("---")
        
        # System info
        st.subheader("System Info")
        model_info = processor.detector.get_model_info()
        st.text(f"Model: {Path(model_info['model_path']).name}")
        st.text(f"Device: {model_info['device']}")
        st.text(f"Classes: {len(model_info['classes'])}")
    
    # Main content area
    if mode == "📸 Image Upload":
        image_upload_mode(processor, save_violations, show_confidence)
    
    elif mode == "🎥 Video Upload":
        video_upload_mode(processor, save_violations)
    
    elif mode == "📹 Webcam (Local)":
        webcam_mode(processor)
    
    elif mode == "📊 Statistics":
        statistics_mode(processor)


def image_upload_mode(processor, save_violations, show_confidence):
    """Image upload and processing mode"""
    st.header("📸 Image Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to detect PPE compliance"
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Display original
            st.image(image, caption="Original Image", use_container_width=True)
            
            # Process button
            if st.button("🔍 Detect PPE", type="primary"):
                with st.spinner("Processing..."):
                    result = processor.process_frame(
                        image_np,
                        save_violation=save_violations,
                        source_name=uploaded_file.name
                    )
                    
                    # Store in session state
                    st.session_state['image_result'] = result
    
    with col2:
        st.subheader("Detection Results")
        
        if 'image_result' in st.session_state:
            result = st.session_state['image_result']
            
            # Display annotated image
            annotated_rgb = cv2.cvtColor(result['annotated_frame'], cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Detected PPE", use_container_width=True)
            
            # Display metrics
            st.markdown("### Detection Summary")
            
            violations = result['violations']['violation_count']
            total_persons = result['violations']['total_persons']
            compliance = result['violations']['compliance_rate']
            
            # Metrics row
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric("Total Persons", total_persons)
            
            with metric_col2:
                st.metric("Violations", violations)
            
            with metric_col3:
                st.metric("Compliance Rate", f"{compliance:.1f}%")
            
            # Alert box
            if violations > 0:
                st.markdown(
                    f'<div class="violation-alert">⚠️ {violations} PPE violation(s) detected!</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="compliant-alert">✅ All workers are compliant!</div>',
                    unsafe_allow_html=True
                )
            
            # Detailed results
            with st.expander("📋 Detailed Detection Results"):
                st.json({
                    'persons_detected': len(result['detections']['persons']),
                    'helmets_detected': len(result['detections']['helmets']),
                    'vests_detected': len(result['detections']['vests']),
                    'violations': violations,
                    'compliance_rate': f"{compliance:.1f}%"
                })
            
            # Violation details
            if violations > 0:
                with st.expander("⚠️ Violation Details"):
                    for i, viol in enumerate(result['violations']['violations'], 1):
                        st.write(f"**Person {i}:**")
                        st.write(f"- Missing: {', '.join(viol['missing_ppe'])}")
                        st.write(f"- Confidence: {viol['person']['conf']:.2%}")
                        st.write("---")


def video_upload_mode(processor, save_violations):
    """Video upload and processing mode"""
    st.header("🎥 Video Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video to detect PPE compliance"
    )
    
    if uploaded_file is not None:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
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
                help="0 = process all frames, 5 = process every 5th frame"
            )
        
        with col2:
            save_output = st.checkbox("Save annotated video", value=False)
        
        if st.button("🔍 Process Video", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            output_path = None
            if save_output:
                output_path = f"output_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            
            try:
                with st.spinner("Processing video... This may take a while."):
                    stats = processor.process_video(
                        video_path,
                        output_path=output_path,
                        display=False,
                        skip_frames=skip_frames
                    )
                
                # Display results
                st.success("✅ Video processing complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Frames Processed", stats['total_frames_processed'])
                with col2:
                    st.metric("Total Violations", stats['total_violations'])
                with col3:
                    st.metric("Frames with Violations", stats['frames_with_violations'])
                with col4:
                    avg_viol = stats['total_violations'] / max(stats['total_frames_processed'], 1)
                    st.metric("Avg Violations/Frame", f"{avg_viol:.2f}")
                
                if save_output and os.path.exists(output_path):
                    st.success(f"Annotated video saved: {output_path}")
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Annotated Video",
                            f,
                            file_name=output_path,
                            mime='video/mp4'
                        )
            
            except Exception as e:
                st.error(f"Error processing video: {e}")
            
            finally:
                # Clean up temp file
                if os.path.exists(video_path):
                    os.unlink(video_path)


def webcam_mode(processor):
    """Webcam mode instructions"""
    st.header("📹 Webcam Mode")
    
    st.info("""
    **Webcam mode runs locally via command line for better performance.**
    
    To use webcam mode:
    
    1. Open a terminal
    2. Navigate to the project directory
    3. Run: `python -m src.webcam_demo`
    
    Or use the inference script:
    ```bash
    python inference.py --source webcam --camera 0
    ```
    
    **Controls:**
    - Press 'q' to quit
    - Press 's' to save current frame
    """)
    
    st.warning("⚠️ Webcam streaming in Streamlit has limited performance. Use the CLI for real-time detection.")
    
    # Could add WebRTC-based streaming here in future


def statistics_mode(processor):
    """Display statistics and violation logs"""
    st.header("📊 System Statistics")
    
    # Get statistics
    stats = processor.get_statistics()
    
    if stats.get('total_frames_logged', 0) == 0:
        st.info("No violations logged yet. Start detecting to see statistics here!")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Violations", stats['total_violations'])
    
    with col2:
        st.metric("Frames Logged", stats['total_frames_logged'])
    
    with col3:
        avg = stats.get('average_violations_per_frame', 0)
        st.metric("Avg Violations/Frame", f"{avg:.2f}")
    
    st.markdown("---")
    
    # Violation logs table
    st.subheader("Recent Violations")
    
    logs = stats.get('logs', [])
    if logs:
        # Convert to DataFrame
        df_data = []
        for log in reversed(logs[-50:]):  # Show last 50
            df_data.append({
                'Timestamp': log['timestamp'],
                'Source': log['source'],
                'Persons': log['total_persons'],
                'Violations': log['violation_count'],
                'Compliance %': f"{log['compliance_rate']:.1f}%",
                'Image': log['image_path']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Download logs
        if st.button("📥 Download Full Logs (JSON)"):
            json_str = json.dumps(logs, indent=2)
            st.download_button(
                "Download",
                json_str,
                file_name=f"violation_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json'
            )
    
    # Violation images gallery
    st.markdown("---")
    st.subheader("Violation Images")
    
    violations_dir = processor.violations_dir
    if os.path.exists(violations_dir):
        image_files = sorted(
            [f for f in os.listdir(violations_dir) if f.endswith(('.jpg', '.png'))],
            reverse=True
        )[:12]  # Show last 12 images
        
        if image_files:
            cols = st.columns(3)
            for i, img_file in enumerate(image_files):
                with cols[i % 3]:
                    img_path = os.path.join(violations_dir, img_file)
                    image = Image.open(img_path)
                    st.image(image, caption=img_file, use_container_width=True)
        else:
            st.info("No violation images found.")


if __name__ == "__main__":
    main()