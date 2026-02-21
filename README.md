# PPE Detection App

PPE (Personal Protective Equipment) Compliance Detection system built with YOLOv8, OpenCV, and Streamlit.

## What This Project Does
- Detects PPE classes from images/video/webcam:
  - `helmet`
  - `vest`
  - `without_helmet`
  - `without_vest`
- Flags violations when `without_helmet` or `without_vest` are detected.
- Saves annotated violation frames to `violations/`.
- Supports:
  - CLI inference (`image`, `video`, `webcam`)
  - Model training with Ultralytics YOLOv8
  - Streamlit dashboard UI (see notes below)

## Project Structure
```text
ppe_detection_app/
|-- app/
|   `-- dashboard.py
|-- config/
|   `-- config.yaml
|-- data/
|   `-- Construction Site.v1i.yolov8/
|       |-- data.yaml
|       |-- train/
|       |-- valid/
|       `-- test/
|-- models/
|   `-- trained/
|       `-- best.pt
|-- src/
|   |-- __main__.py
|   |-- detector.py
|   |-- inference.py
|   |-- processor.py
|   |-- train.py
|   |-- utils.py
|   `-- violation_logic.py
|-- requirements.txt
|-- pyproject.toml
|-- run_inference.bat
`-- README.md
```

## Requirements
- Python 3.8+
- `pip`
- Optional GPU: CUDA-compatible PyTorch

## Installation
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration
Main config file: `config/config.yaml`

Key settings:
- `model.weights_path`: model file path (default: `models/trained/best.pt`)
- `model.confidence_threshold`: detection confidence threshold
- `model.iou_threshold`: YOLO NMS IoU threshold
- `model.device`: `cpu` or `cuda`
- `storage.violations_dir`: where violation images are stored
- `storage.save_violations`: enable/disable saving violation frames

## Run Inference (CLI)
The main entry point is `src/inference.py`.

### 1) Image (single file)
```bash
python -m src.inference --source image --input path/to/image.jpg --output path/to/result.jpg
```

### 2) Image folder (batch)
```bash
python -m src.inference --source image --input path/to/images --output path/to/output_folder
```

### 3) Video
```bash
python -m src.inference --source video --input path/to/video.mp4 --output path/to/output.mp4
```

### 4) Webcam
```bash
python -m src.inference --source webcam --camera 0
```

Useful flags:
```bash
--display
--skip-frames 5
--model models/trained/best.pt
--config config/config.yaml
--no-save-violations
```

### Windows shortcut
```bat
run_inference.bat image "path/to/images" "path/to/output"
run_inference.bat video "input.mp4" "output.mp4"
run_inference.bat webcam
```

## Train the Model
Training script: `src/train.py`

Validate dataset only:
```bash
python src/train.py --validate-only
```

Train:
```bash
python src/train.py \
  --data "data/Construction Site.v1i.yolov8/data.yaml" \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device cpu \
  --project models \
  --name ppe_detection
```

After training, copy best weights to:
- `models/trained/best.pt`

## Dashboard (Streamlit)
Run:
```bash
streamlit run app/dashboard.py
```

Note:
- The current dashboard code has fields that are out of sync with the active inference pipeline.
- For stable operation, use the CLI commands above.

## Dataset
This repo includes a YOLO-format dataset in:
- `data/Construction Site.v1i.yolov8/`

Classes from `data.yaml`:
- `helmet`
- `vest`
- `without_helmet`
- `without_vest`

## Packaging Commands (optional)
Using `pyproject.toml`:
```bash
pip install -e .
```

Then available console scripts:
```bash
ppe-detect
ppe-train
ppe-dashboard
```

## Troubleshooting
- Model not found:
  - Ensure `models/trained/best.pt` exists.
  - Or pass `--model` explicitly.
- Camera not opening:
  - Check webcam index (`--camera 0`, `--camera 1`, etc.).
- Slow inference:
  - Increase `--skip-frames` for video.
  - Use `model.device: "cuda"` if GPU is available.
- OpenCV window not appearing:
  - Run from local desktop session (not headless terminal).

## License
Dataset metadata indicates Roboflow export with `CC BY 4.0`.
Add your project license details here if needed.
