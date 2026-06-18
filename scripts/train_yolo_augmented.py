"""Train YOLO on augmented full images dataset."""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    # Load pretrained model
    model = YOLO(PROJECT_ROOT / "runs/detect/yolo_cropped_v2/weights/best.pt")
    
    print("Training YOLO on augmented full images...")
    print("Dataset: 310 images (augmented from 31)")
    
    results = model.train(
        data=str(PROJECT_ROOT / "data/augmented_full/data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        patience=15,
        save=True,
        project=str(PROJECT_ROOT / "runs/detect"),
        name="yolo_augmented_v1",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.3,
        mosaic=0.8,
        mixup=0.1,
        close_mosaic=10,
        workers=4,
        device="cpu",
        verbose=True,
    )
    
    print(f"\nTraining complete!")
    best_path = PROJECT_ROOT / "runs/detect/yolo_augmented_v1/weights/best.pt"
    print(f"Best model: {best_path}")
    
    return best_path


if __name__ == "__main__":
    main()
