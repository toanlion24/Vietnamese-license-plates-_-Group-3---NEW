"""Quick training with fewer epochs for testing."""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    model = YOLO(PROJECT_ROOT / "weights/yolov8_license_plate.pt")
    
    results = model.train(
        data=str(PROJECT_ROOT / "data/yolo_dataset_full/data.yaml"),
        epochs=10,  # Reduced for quick test
        imgsz=640,  # Smaller images for faster training
        batch=2,
        patience=5,
        save=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="yolo_quick",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.01,
        close_mosaic=5,
        workers=2,
        device="cpu",
        verbose=True,
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: {PROJECT_ROOT / 'runs/detect/yolo_quick/weights/best.pt'}")


if __name__ == "__main__":
    main()
