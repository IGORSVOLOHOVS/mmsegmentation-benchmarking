import os
import argparse
import numpy as np
from PIL import Image
import torch
import mmcv
from mmseg.visualization import SegLocalVisualizer
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

def calculate_dice(pred, gt, num_classes):
    dices = []
    # Ignore background (class 0)
    for cls in range(1, num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = pred_cls.sum() + gt_cls.sum()
        if union == 0:
            if intersection == 0:
                dices.append(1.0)
            else:
                dices.append(0.0)
        else:
            dices.append(2. * intersection / union)
    return np.mean(dices) if len(dices) > 0 else 0.0

def parse_args():
    parser = argparse.ArgumentParser(description='Save best and worst predictions')
    parser.add_argument('pred_dir', help='Predicted masks directory')
    parser.add_argument('gt_dir', help='Ground truth masks directory')
    parser.add_argument('img_dir', help='Original images directory')
    parser.add_argument('out_dir', help='Output directory for best/worst overlays')
    parser.add_argument('--num-classes', type=int, default=3, help='Number of classes')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    best_dir = os.path.join(args.out_dir, 'best')
    worst_dir = os.path.join(args.out_dir, 'worst')
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(worst_dir, exist_ok=True)
    
    img_list = [f for f in os.listdir(args.pred_dir) if f.endswith('.png')]
    scores = []
    
    print("Calculating Dice scores...")
    for img_name in img_list:
        pred_path = os.path.join(args.pred_dir, img_name)
        gt_path = os.path.join(args.gt_dir, img_name)
        
        pred = np.array(Image.open(pred_path))
        gt = np.array(Image.open(gt_path))
        
        dice = calculate_dice(pred, gt, args.num_classes)
        scores.append((dice, img_name))
        
    scores.sort(key=lambda x: x[0])
    
    # Init visualizer
    visualizer = SegLocalVisualizer(
        vis_backends=[dict(type='LocalVisBackend')],
        save_dir=args.out_dir,
        alpha=0.5)
    visualizer.dataset_meta = dict(
        classes=('background', 'cat', 'dog'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]])
    
    # Process worst 5
    print("Saving worst 5 examples...")
    for idx, (score, img_name) in enumerate(scores[:5]):
        img_path = os.path.join(args.img_dir, img_name.replace('.png', '.jpg'))
        img = mmcv.imread(img_path, channel_order='rgb')
        pred = np.array(Image.open(os.path.join(args.pred_dir, img_name)))
        gt = np.array(Image.open(os.path.join(args.gt_dir, img_name)))

        data_sample = SegDataSample()
        data_sample.pred_sem_seg = PixelData(data=torch.from_numpy(pred).unsqueeze(0))
        data_sample.gt_sem_seg = PixelData(data=torch.from_numpy(gt).unsqueeze(0))

        visualizer.add_datasample(
            f'worst_{idx}_{score:.4f}_{img_name}',
            img,
            data_sample,
            draw_gt=True,
            draw_pred=True,
            show=False,
            out_file=os.path.join(worst_dir, f'worst_{idx}_score_{score:.4f}_{img_name}')
        )
        
    # Process best 5
    print("Saving best 5 examples...")
    # Best ones usually are scores[-1], scores[-2], etc. So [::-1]
    best_scores = scores[-5:][::-1]
    for idx, (score, img_name) in enumerate(best_scores):
        img_path = os.path.join(args.img_dir, img_name.replace('.png', '.jpg'))
        img = mmcv.imread(img_path, channel_order='rgb')
        pred = np.array(Image.open(os.path.join(args.pred_dir, img_name)))
        gt = np.array(Image.open(os.path.join(args.gt_dir, img_name)))

        data_sample = SegDataSample()
        data_sample.pred_sem_seg = PixelData(data=torch.from_numpy(pred).unsqueeze(0))
        data_sample.gt_sem_seg = PixelData(data=torch.from_numpy(gt).unsqueeze(0))

        visualizer.add_datasample(
            f'best_{idx}_{score:.4f}_{img_name}',
            img,
            data_sample,
            draw_gt=True,
            draw_pred=True,
            show=False,
            out_file=os.path.join(best_dir, f'best_{idx}_score_{score:.4f}_{img_name}')
        )

if __name__ == '__main__':
    main()
