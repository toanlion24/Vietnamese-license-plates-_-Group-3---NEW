"""Train YOLO on old cropped plates dataset with augmentation."""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    # Load pre-trained model
    model = YOLO(PROJECT_ROOT / "weights/yolov8_license_plate.pt")
    
    print("Training YOLO on cropped plates dataset...")
    print("Dataset: 3204 images (cropped plates)")
    
    # Train with better settings
    results = model.train(
        data=str(PROJECT_ROOT / "data/data.yaml"),  # Original config
        epochs=30,
        imgsz=640,
        batch=16,
        patience=10,
        save=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="yolo_cropped_v2",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,  # Enable augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        close_mosaic=10,
        workers=4,
        device="cpu",
        verbose=True,
    )
    
    print(f"\nTraining complete!")
    best_path = PROJECT_ROOT / "runs/detect/yolo_cropped_v2/weights/best.pt"
    print(f"Best model: {best_path}")
    
    return best_path


if __name__ == "__main__":
    main()
