import os
import cv2
import numpy as np
from tqdm import tqdm

def generate_labeled_dataset(image_path: str, label_path: str, output_path: str, colormap: dict = None, alpha: float = 0.5):
    """
    Blends segmentation masks into images with specific colors and saves the results.
    Args:
        image_path: Directory with images.
        label_path: Directory with masks.
        output_path: Directory to save blended images.
        colormap: Optional dictionary mapping class ID to BGR color [B, G, R].
                  Default: 1: [255, 0, 0] (Blue), 2: [0, 255, 0] (Green).
        alpha: Transparency factor for the mask (0 to 1).
    """
    if colormap is None:
        colormap = {
            1: [255, 0, 0], # Blue
            2: [0, 255, 0]  # Green
        }
        
    os.makedirs(output_path, exist_ok=True)
    images = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Generating colored labeled images in {output_path}...")
    for img_name in tqdm(images):
        lbl_name = os.path.splitext(img_name)[0] + '.png'
        img_file = os.path.join(image_path, img_name)
        lbl_file = os.path.join(label_path, lbl_name)
        
        if not os.path.exists(lbl_file):
            continue
            
        img = cv2.imread(img_file)
        # Read mask as 1-channel (grayscale)
        lbl = cv2.imread(lbl_file, cv2.IMREAD_GRAYSCALE)
        
        if img is None or lbl is None:
            continue

        if img.shape[:2] != lbl.shape[:2]:
            lbl = cv2.resize(lbl, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Create a color mask
        color_mask = np.zeros_like(img)
        for class_id, color in colormap.items():
            color_mask[lbl == class_id] = color
            
        # Blend: img * (1) + color_mask * alpha
        res = cv2.addWeighted(img, 1.0, color_mask, alpha, 0)

        cv2.imwrite(os.path.join(output_path, lbl_name), res)