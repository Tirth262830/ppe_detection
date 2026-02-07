"""
YOLO Model Training Script for PPE Detection
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import yaml
import argparse
import torch


def train_ppe_model(
    data_yaml: str,
    model_type: str = 'yolov8n.pt',
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    project: str = 'models',
    name: str = 'ppe_detection',
    device: str = 'cpu',
    patience: int = 50,
    save_period: int = 10,
    workers: int = 2
):
    """
    Train YOLO model for PPE detection
    
    Args:
        data_yaml: Path to data.yaml file
        model_type: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size for training
        project: Project directory to save runs
        name: Run name
        device: Device to use (cpu, cuda, 0, 1, etc.)
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        workers: Number of dataloader workers
    """
    
    print("=" * 80)
    print("PPE Detection Model Training")
    print("=" * 80)
    
    # Check if data.yaml exists
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")
    
    # Load and verify data.yaml
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\nDataset Configuration:")
    print(f"  Train images: {data_config.get('train', 'Not specified')}")
    print(f"  Validation images: {data_config.get('val', 'Not specified')}")
    print(f"  Test images: {data_config.get('test', 'Not specified')}")
    print(f"  Number of classes: {data_config.get('nc', 'Not specified')}")
    print(f"  Class names: {data_config.get('names', 'Not specified')}")
    
    # Check CUDA availability
    if device != 'cpu':
        if torch.cuda.is_available():
            print(f"\n✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Using device: {device}")
        else:
            print("\n⚠ CUDA not available, falling back to CPU")
            device = 'cpu'
    else:
        print(f"\n  Using device: CPU")
    
    # Initialize YOLO model
    print(f"\nInitializing YOLO model: {model_type}")
    model = YOLO(model_type)
    
    # Training configuration
    print("\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Image size: {imgsz}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    print(f"  Early stopping patience: {patience}")
    print(f"  Save period: {save_period}")
    print(f"  Workers: {workers}")
    
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        workers=workers,
        plots=True,  # Save training plots
        save=True,   # Save checkpoints
        verbose=True,
        exist_ok=False,  # Don't overwrite existing runs
        pretrained=True,  # Use pretrained weights
        optimizer='auto',
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,  # Cosine learning rate scheduler
        label_smoothing=0.0,
        val=True  # Validate during training
    )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
    
    # Print results location
    save_dir = Path(project) / name
    print(f"\nResults saved to: {save_dir}")
    print(f"Best model: {save_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {save_dir / 'weights' / 'last.pt'}")
    
    # Print metrics
    print("\nFinal Metrics:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 'N/A')}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 'N/A')}")
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"\nTo use the trained model:")
    print(f"  1. Copy {save_dir / 'weights' / 'best.pt'} to models/trained/best.pt")
    print(f"  2. Update config.yaml with the correct model path")
    print(f"  3. Run inference using the dashboard or CLI")
    
    return results


def validate_dataset(data_yaml: str):
    """Validate dataset structure and configuration"""
    print("Validating dataset...")
    
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required fields
    required = ['train', 'val', 'nc', 'names']
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field in data.yaml: {field}")
    
    # Check if paths exist
    base_path = Path(data_yaml).parent
    
    for split in ['train', 'val', 'test']:
        if split in config:
            split_path = base_path / config[split]
            if not split_path.exists():
                print(f"⚠ Warning: {split} path does not exist: {split_path}")
            else:
                print(f"✓ Found {split} data: {split_path}")
    
    # Verify number of classes matches names
    if len(config['names']) != config['nc']:
        raise ValueError(
            f"Number of classes ({config['nc']}) doesn't match "
            f"number of class names ({len(config['names'])})"
        )
    
    print(f"✓ Dataset validation passed")
    print(f"  Classes ({config['nc']}): {config['names']}")


def main():
    """Main training function with CLI"""
    parser = argparse.ArgumentParser(description='Train YOLO model for PPE detection')
    
    parser.add_argument(
        '--data',
        type=str,
        default='data/Construction Site.v1i.yolov8/data.yaml',
        help='Path to data.yaml file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
        help='YOLO model variant (n=nano, s=small, m=medium, l=large, x=xlarge)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size (-1 for auto-batch)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu, cuda, 0, 1, etc.)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='models',
        help='Project directory'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='ppe_detection',
        help='Run name'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=2,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate dataset without training'
    )
    
    args = parser.parse_args()
    
    try:
        # Validate dataset
        validate_dataset(args.data)
        
        if args.validate_only:
            print("\n✓ Dataset validation complete. Exiting (--validate-only flag set)")
            return
        
        # Train model
        train_ppe_model(
            data_yaml=args.data,
            model_type=args.model,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch,
            project=args.project,
            name=args.name,
            device=args.device,
            patience=args.patience,
            workers=args.workers
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()