"""Create synthetic test images for demo and testing."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_synthetic_plate_image(plate_text: str, output_path: str):
    """Create a synthetic image with a license plate."""
    # Create background (simulate road scene)
    img = Image.new('RGB', (640, 480), color=(70, 70, 70))
    draw = ImageDraw.Draw(img)
    
    # Add some noise to simulate real scene
    arr = np.array(img)
    noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    
    # Draw plate background (white with border)
    plate_x, plate_y = 170, 200
    plate_w, plate_h = 300, 80
    
    # Plate border (blue for VN plates)
    draw.rectangle([plate_x-3, plate_y-3, plate_x+plate_w+3, plate_y+plate_h+3], 
                   fill=(0, 0, 255))
    # Plate background (white)
    draw.rectangle([plate_x, plate_y, plate_x+plate_w, plate_y+plate_h], 
                   fill=(255, 255, 255))
    
    # Draw plate text
    try:
        font_size = 48
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Calculate text position to center it
    bbox = draw.textbbox((0, 0), plate_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    text_x = plate_x + (plate_w - text_w) // 2
    text_y = plate_y + (plate_h - text_h) // 2 - 5
    
    # Draw text shadow
    draw.text((text_x+2, text_y+2), plate_text, fill=(100, 100, 100), font=font)
    # Draw text
    draw.text((text_x, text_y), plate_text, fill=(0, 0, 0), font=font)
    
    # Save
    img.save(output_path)
    print(f"Created: {output_path}")

def create_sample_images():
    """Create multiple sample images with different plates."""
    output_dir = "data/samples"
    os.makedirs(output_dir, exist_ok=True)
    
    plates = [
        "30G12345",  # Standard car plate
        "51H04321",  # With potential OCR issues
        "43A12345",  # Different province
        "29D98765",  # Another province
        "47K56789",  # Bike plate style
    ]
    
    for i, plate in enumerate(plates):
        output_path = os.path.join(output_dir, f"sample_{i+1}_{plate}.jpg")
        create_synthetic_plate_image(plate, output_path)
    
    print(f"\nCreated {len(plates)} sample images in {output_dir}/")

if __name__ == "__main__":
    create_sample_images()
