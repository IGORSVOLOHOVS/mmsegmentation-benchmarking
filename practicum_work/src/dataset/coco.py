import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

def generate_coco_dataset(img_dir: str, lbl_dir: str, output_path: str, categories: list = None):
    """
    Converts segmentation masks to COCO JSON format.
    Args:
        img_dir: Directory with images.
        lbl_dir: Directory with mask labels (.png).
        output_path: Path to save the annotations.json.
        categories: List of category names.
    """
    if categories is None:
        categories = ["background", "class1", "class2"] # Default based on what I saw [0, 1, 2]
    
    coco = {
        "info": {
            "description": "Custom COCO Dataset",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "User",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(categories)]
    }
    
    imgs = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    ann_id = 1
    
    print(f"Generating COCO dataset for {len(imgs)} images...")
    for i, img_name in enumerate(tqdm(imgs)):
        img_path = os.path.join(img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        height, width = image.shape[:2]
        img_id = i + 1
        
        coco["images"].append({
            "id": img_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })
        
        lbl_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(lbl_dir, lbl_name)
        
        if not os.path.exists(mask_path):
            continue
            
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # For each class (excluding background 0 if desired, but COCO usually starts from 1 for objects)
        # Assuming class 0 is background, we start from 1.
        for cat_id in range(1, len(categories)):
            cls_mask = (mask == cat_id).astype(np.uint8)
            if np.sum(cls_mask) == 0:
                continue
                
            contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) < 3:
                    continue
                
                segmentation = contour.flatten().tolist()
                # Ensure we have at least 6 points (3 pairs)
                if len(segmentation) < 6:
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": [segmentation],
                    "area": float(area),
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "iscrowd": 0
                })
                ann_id += 1
                
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)
    
    print(f"Saved {len(coco['annotations'])} annotations to {output_path}")