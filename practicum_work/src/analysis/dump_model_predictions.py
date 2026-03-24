import os
import argparse
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
import numpy as np
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Dump model predictions')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('img_dir', help='Image directory (e.g. test images)')
    parser.add_argument('out_dir', help='Output directory for predictions')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the model
    model = init_model(args.config, args.checkpoint, device='cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    img_list = [f for f in os.listdir(args.img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    print(f"Running inference on {len(img_list)} images...")
    for img_name in tqdm(img_list):
        img_path = os.path.join(args.img_dir, img_name)
        result = inference_model(model, img_path)
        
        # Get the predicted segmentation mask
        pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy()
        
        # Save as PNG
        out_name = os.path.splitext(img_name)[0] + '.png'
        out_path = os.path.join(args.out_dir, out_name)
        Image.fromarray(pred_sem_seg.astype(np.uint8)).save(out_path)
        
    print(f"Predictions dumped to {args.out_dir}")

if __name__ == '__main__':
    main()
