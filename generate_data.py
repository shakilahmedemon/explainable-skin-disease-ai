"""
Generate synthetic skin lesion images for initial model training
This addresses the core issue by providing sample data to train the models
"""
import os
import numpy as np
import cv2
from PIL import Image
import argparse
from pathlib import Path
import random

def create_synthetic_skin_lesion(class_name, size=(224, 224)):
    """
    Create a synthetic skin lesion image based on the class
    """
    # Create a base skin-colored image
    img = np.random.randint(240, 255, (*size, 3), dtype=np.uint8)
    
    # Add some skin texture variations
    for _ in range(50):
        x = random.randint(0, size[0]-1)
        y = random.randint(0, size[1]-1)
        radius = random.randint(1, 3)
        intensity = random.randint(200, 255)
        cv2.circle(img, (x, y), radius, (intensity,)*3, -1)
    
    # Define characteristics for different skin conditions
    if class_name == "melanoma":
        # Irregular, asymmetrical dark spot with multiple colors
        center_x = size[0] // 2 + random.randint(-30, 30)
        center_y = size[1] // 2 + random.randint(-30, 30)
        irregular_shape = np.zeros(size, dtype=np.uint8)
        
        # Create an irregular shape
        for _ in range(20):
            angle = random.uniform(0, 2*np.pi)
            distance = random.randint(20, 50)
            x = int(center_x + distance * np.cos(angle))
            y = int(center_y + distance * np.sin(angle))
            radius = random.randint(8, 15)
            cv2.circle(irregular_shape, (x, y), radius, 255, -1)
        
        # Apply the shape to the image with different colors (brown, black, red)
        mask = irregular_shape > 0
        colors = [(50, 30, 25), (30, 20, 15), (80, 40, 30), (100, 30, 40)]  # Various lesion colors
        for i in range(size[0]):
            for j in range(size[1]):
                if mask[i, j]:
                    color = random.choice(colors)
                    # Add some variation, ensuring values stay within uint8 bounds
                    new_color = []
                    for c in color:
                        new_val = c + random.randint(-20, 20)
                        new_val = max(0, min(255, new_val))  # Clamp to valid range
                        new_color.append(new_val)
                    img[i, j] = tuple(new_color)
    
    elif class_name == "basal_cell_carcinoma":
        # Shiny, pearly bump
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = random.randint(30, 60)
        cv2.circle(img, (center_x, center_y), radius, (200, 180, 170), -1)  # Pearly color
        
        # Add shine effect
        shine_x, shine_y = center_x - 10, center_y - 10
        cv2.circle(img, (shine_x, shine_y), radius//4, (240, 240, 240), -1)
    
    elif class_name == "squamous_cell_carcinoma":
        # Red, scaly patch
        center_x, center_y = size[0] // 2, size[1] // 2
        for _ in range(15):
            x = center_x + random.randint(-50, 50)
            y = center_y + random.randint(-50, 50)
            radius = random.randint(10, 25)
            # Reddish color
            cv2.circle(img, (x, y), radius, (150, 80, 70), -1)
    
    elif class_name == "nevus":
        # Regular, symmetrical mole
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = random.randint(20, 40)
        # Brown mole color
        cv2.circle(img, (center_x, center_y), radius, (80, 50, 40), -1)
        
        # Add slight shading
        cv2.circle(img, (center_x-5, center_y-5), radius//2, (100, 70, 60), -1)
    
    elif class_name == "dermatofibroma":
        # Firm, round bump
        center_x, center_y = size[0] // 2, size[1] // 2
        radius = random.randint(15, 35)
        # Brownish color
        cv2.circle(img, (center_x, center_y), radius, (120, 90, 70), -1)
    
    elif class_name == "vascular_lesion":
        # Red/purple vascular pattern
        center_x, center_y = size[0] // 2, size[1] // 2
        for _ in range(8):
            x = center_x + random.randint(-40, 40)
            y = center_y + random.randint(-40, 40)
            radius = random.randint(5, 15)
            # Red/purple color
            cv2.circle(img, (x, y), radius, (120, 40, 80), -1)
    
    elif class_name == "actinic_keratosis":
        # Rough, scaly patch
        for _ in range(25):
            x = random.randint(50, size[0]-50)
            y = random.randint(50, size[1]-50)
            radius = random.randint(5, 12)
            # Whitish/scaly color
            cv2.circle(img, (x, y), radius, (180, 160, 150), -1)
    
    return img

def generate_dataset(num_samples_per_class=50, output_dir="data/train"):
    """
    Generate synthetic dataset for all classes
    """
    classes = [
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic dataset in: {output_path}")
    print(f"Classes: {classes}")
    print(f"Samples per class: {num_samples_per_class}")
    
    total_generated = 0
    
    for class_name in classes:
        class_path = output_path / class_name
        class_path.mkdir(exist_ok=True)
        
        print(f"\nGenerating {num_samples_per_class} samples for {class_name}...")
        
        for i in range(num_samples_per_class):
            # Create synthetic image
            img = create_synthetic_skin_lesion(class_name)
            
            # Add some random transformations to increase variety
            if random.random() > 0.5:
                # Random rotation
                angle = random.randint(-30, 30)
                h, w = img.shape[:2]
                matrix = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                img = cv2.warpAffine(img, matrix, (w, h))
            
            if random.random() > 0.5:
                # Random flip
                if random.random() > 0.5:
                    img = cv2.flip(img, 1)  # Horizontal flip
                else:
                    img = cv2.flip(img, 0)  # Vertical flip
            
            # Save image
            img_pil = Image.fromarray(img)
            filename = f"{class_name}_{i+1:03d}.png"
            filepath = class_path / filename
            img_pil.save(filepath)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_samples_per_class} images for {class_name}")
        
        total_generated += num_samples_per_class
        print(f"✅ Completed {class_name}: {num_samples_per_class} images")
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"Total images generated: {total_generated}")
    print(f"Dataset location: {output_path}")
    print(f"Average per class: {num_samples_per_class}")

def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic Skin Lesion Dataset')
    parser.add_argument('--samples-per-class', type=int, default=50,
                        help='Number of samples to generate per class (default: 50)')
    parser.add_argument('--output-dir', type=str, default='data/train',
                        help='Output directory for generated dataset (default: data/train)')
    
    args = parser.parse_args()
    
    print("🔬 GENERATING SYNTHETIC SKIN LESION DATASET")
    print("="*50)
    
    generate_dataset(
        num_samples_per_class=args.samples_per_class,
        output_dir=args.output_dir
    )
    
    print("\n💡 Next steps:")
    print(f"  1. Review generated images in {args.output_dir}")
    print("  2. Train the model: python train.py")
    print("  3. Test the application: streamlit run app/app.py")

if __name__ == "__main__":
    main()
