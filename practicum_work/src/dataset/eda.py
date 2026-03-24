import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def missed_labels(img_dir: str, lbl_dir: str):
    imgs = os.listdir(img_dir)
    missing = []
    for img in imgs:
        lbl_name = os.path.splitext(img)[0] + '.png'
        if not os.path.exists(os.path.join(lbl_dir, lbl_name)):
            missing.append(img)
    
    if missing:
        print(f"Missed labels for {len(missing)} images: {missing[:10]}...")
    else:
        print("No missing labels found.")
    return missing

def class_distribution(lbl_dir: str, categories: list = None):
    labels = os.listdir(lbl_dir)
    total_counts = {}
    
    # print("Calculating class distribution...")
    for lbl in tqdm(labels):
        lbl_path = os.path.join(lbl_dir, lbl)
        mask = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        unique, counts = np.unique(mask, return_counts=True)
        for u, c in zip(unique, counts):
            total_counts[int(u)] = total_counts.get(int(u), 0) + int(c)
            
    print("Class distribution (pixel counts):")
    for cls, count in sorted(total_counts.items()):
        name = categories[cls] if categories and cls < len(categories) else f"Class {cls}"
        print(f"{name}: {count} pixels")
    
    # Plotting
    if categories:
        names = categories
        counts = [total_counts.get(i, 0) for i in range(len(categories))]
    else:
        names = [f"Class {i}" for i in sorted(total_counts.keys())]
        counts = [total_counts[i] for i in sorted(total_counts.keys())]
        
    plt.figure(figsize=(10, 6))
    plt.bar(names, counts, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Pixel Count')
    plt.title('Class Distribution')
    plt.show()
    
    return total_counts

def size_distribution(img_dir: str):
    imgs = os.listdir(img_dir)
    sizes = []
    
    # print("Calculating size distribution...")
    for img_name in tqdm(imgs):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        sizes.append(img.shape[:2])
        
    sizes = np.array(sizes)
    h, w = sizes[:, 0], sizes[:, 1]
    
    print(f"Average size: {np.mean(h):.2f}x{np.mean(w):.2f}")
    print(f"Min size: {np.min(h)}x{np.min(w)}")
    print(f"Max size: {np.max(h)}x{np.max(w)}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(h, w, alpha=0.5)
    plt.xlabel('Height')
    plt.ylabel('Width')
    plt.title('Image Size Distribution')
    plt.grid(True)
    plt.show()
    
    return sizes
