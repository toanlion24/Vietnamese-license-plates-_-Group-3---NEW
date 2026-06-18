"""
Augment 39 annotated images to create larger dataset for training.
"""

from pathlib import Path
import cv2
import numpy as np
import random
import shutil
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "data/yolo_full_train"
OUTPUT_DIR = PROJECT_ROOT / "data/augmented_full"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def augment_image(image, bbox, aug_type):
    """Apply augmentation to image and adjust bbox."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    
    if aug_type == "flip_horizontal":
        img = cv2.flip(image, 1)
        new_x = w - x - bw
        return img, (new_x, y, bw, bh)
    
    elif aug_type == "brightness_up":
        img = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
        return img, bbox
    
    elif aug_type == "brightness_down":
        img = cv2.convertScaleAbs(image, alpha=0.8, beta=-20)
        return img, bbox
    
    elif aug_type == "blur":
        img = cv2.GaussianBlur(image, (5, 5), 0)
        return img, bbox
    
    elif aug_type == "noise":
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        img = cv2.add(image, noise)
        return img, bbox
    
    elif aug_type == "rotate_small":
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        img = cv2.warpAffine(image, M, (w, h))
        
        # Rotate bbox
        cx, cy = x + bw/2, y + bh/2
        new_cx = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        new_cy = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        return img, (int(new_cx - bw/2), int(new_cy - bh/2), bw, bh)
    
    elif aug_type == "contrast":
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img, bbox
    
    elif aug_type == "shadow":
        img = image.copy()
        overlay = img.copy()
        pts = np.array([[random.randint(0, w//2), random.randint(0, h)],
                        [random.randint(w//2, w), random.randint(0, h)],
                        [random.randint(w//2, w), h],
                        [random.randint(0, w//2), h]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 0, 0))
        img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        return img, bbox
    
    elif aug_type == "perspective":
        # Slight perspective transform
        pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        offset = random.randint(-20, 20)
        pts2 = np.float32([[offset, offset], [w-offset, 0], [0, h-offset], [w, h-offset]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(image, M, (w, h))
        return img, bbox
    
    return image, bbox


def create_augmented_dataset():
    """Create augmented dataset from 39 images."""
    
    print("=" * 60)
    print("AUGMENT DATASET")
    print("=" * 60)
    
    # Get source images and labels
    images_dir = INPUT_DIR / "images/train"
    labels_dir = INPUT_DIR / "labels/train"
    
    if not images_dir.exists():
        print(f"Error: Images dir not found: {images_dir}")
        return
    
    image_files = sorted(images_dir.glob("*.jpg"))
    print(f"\nFound {len(image_files)} source images")
    
    # Augmentation types
    aug_types = [
        "flip_horizontal",
        "brightness_up", 
        "brightness_down",
        "blur",
        "noise",
        "rotate_small",
        "contrast",
        "shadow",
        "perspective"
    ]
    
    # Output directories
    out_train = OUTPUT_DIR / "train"
    out_images = out_train / "images"
    out_labels = out_train / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)
    
    out_val = OUTPUT_DIR / "val"
    out_val_images = out_val / "images"
    out_val_labels = out_val / "labels"
    out_val_images.mkdir(parents=True, exist_ok=True)
    out_val_labels.mkdir(parents=True, exist_ok=True)
    
    # Augment each image 5 times (39 * 6 = 234 images)
    count = 0
    annotations = []
    
    for img_path in image_files:
        stem = img_path.stem
        
        # Load image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Load label
        label_path = labels_dir / f"{stem}.txt"
        if not label_path.exists():
            print(f"Warning: No label for {stem}")
            continue
        
        with open(label_path) as f:
            parts = f.read().strip().split()
            if len(parts) >= 5:
                # YOLO format to pixel
                cls, cx, cy, bw, bh = map(float, parts[:5])
                cx_px = int(cx * w)
                cy_px = int(cy * h)
                bw_px = int(bw * w)
                bh_px = int(bh * h)
                bbox = (cx_px - bw_px//2, cy_px - bh_px//2, bw_px, bh_px)
        
        # Save original
        aug_name = "original"
        out_path = out_images / f"{stem}_{aug_name}.jpg"
        cv2.imwrite(str(out_path), img)
        
        # Save label
        with open(label_path) as lf:
            label_content = lf.read()
        with open(out_labels / f"{stem}_{aug_name}.txt", "w") as f:
            f.write(label_content)
        
        count += 1
        annotations.append({"file": f"{stem}_{aug_name}.jpg", "source": stem, "aug": aug_name})
        
        # Apply augmentations
        for aug_type in aug_types:
            aug_img, aug_bbox = augment_image(img.copy(), bbox, aug_type)
            
            aug_filename = f"{stem}_{aug_type}"
            out_path = out_images / f"{aug_filename}.jpg"
            cv2.imwrite(str(out_path), aug_img)
            
            # Convert bbox back to YOLO
            x, y, bw, bh = aug_bbox
            x = max(0, x)
            y = max(0, y)
            ncx = (x + bw/2) / w
            ncy = (y + bh/2) / h
            nbw = bw / w
            nbh = bh / h
            
            with open(out_labels / f"{aug_filename}.txt", "w") as f:
                f.write(f"0 {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}\n")
            
            count += 1
            annotations.append({"file": f"{aug_filename}.jpg", "source": stem, "aug": aug_type})
    
    # Split train/val (90/10)
    random.seed(42)
    random.shuffle(annotations)
    split_idx = int(len(annotations) * 0.9)
    train_ann = annotations[:split_idx]
    val_ann = annotations[split_idx:]
    
    # Move some to val
    val_count = 0
    for ann in val_ann:
        src_img = out_images / ann["file"]
        src_label = out_labels / ann["file"].replace(".jpg", ".txt")
        
        if src_img.exists():
            shutil.move(str(src_img), str(out_val_images / ann["file"]))
        if src_label.exists():
            shutil.move(str(src_label), str(out_val_labels / ann["file"].replace(".jpg", ".txt")))
        val_count += 1
    
    print(f"\nDataset created: {OUTPUT_DIR}")
    print(f"  Train images: {len(train_ann)}")
    print(f"  Val images: {len(val_ann)}")
    print(f"  Total: {len(annotations)}")
    
    # Create train.txt
    with open(OUTPUT_DIR / "train.txt", "w") as f:
        for ann in train_ann:
            f.write(f"{out_train / 'images' / ann['file']}\n")
    
    # Create data.yaml
    yaml = f"""path: {OUTPUT_DIR}
train: train.txt
val: val/images
names:
  0: plate
nc: 1
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml)
    
    print(f"\ndata.yaml created")
    
    return OUTPUT_DIR


if __name__ == "__main__":
    create_augmented_dataset()
