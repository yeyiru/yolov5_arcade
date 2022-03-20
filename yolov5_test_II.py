
import os
import cv2
import wandb
import torch
import argparse
import pandas as pd
import numpy as np

from utils.metrics_w_iou import w_iou
from utils.general import xywh2xyxy
import val_for_lrp


if __name__ == '__main__':
    
    hyp = dict(
        mode1 = 'coco', # 'nopre', 'coco', 'beauty'
        mode2 = 'merge', # 'small', 'merge'
        mode3 = 's',
        idx = 0,          # 0, 1, 2, 3, 4
        iou_t = 0.8)
    run = wandb.init(project='yolov5_val', config=hyp)
    hyp = wandb.config
    if hyp['mode2'] != 'merge':
        gt_yaml = '/data/Arcade_dataset/arcade/arcade.yaml'
    else:
        gt_yaml = '/data/Arcade_dataset/arcade_merge/arcade.yaml'
    
    mode = hyp['mode3'] + '_' + hyp['mode1'] + '_' + hyp['mode2']
    pt_dir = os.path.join('./artifacts/', mode)
    pts = os.listdir(pt_dir)
    pts.sort()
    pt = pts[int(hyp['idx'])]
    nms_iou = float(pt.split('_')[-1].replace('.pt', ''))
    pt = os.path.join(pt_dir, pt)

    opt = dict(
        data=gt_yaml,
        test_iou_thres=hyp['iou_t'],
        weights=pt,
        conf_thres=0.001,
        iou_thres=nms_iou,
        task='train')
    # opt = argparse.Namespace(**opt)
    tu, maps, t, log = val_for_lrp.run(**opt)
    wandb.log({'ap.5.95.05': tu[3]})
    log.to_csv(f"log_csv/{'_'.join([hyp['mode1'], hyp['mode2'], str(hyp['idx'])])}.csv", index=0)