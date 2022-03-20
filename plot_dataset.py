import os
import torch
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.general import xywh2xyxy
from utils.plots import colors, plot_one_box

label_dir = '/data/Arcade_dataset/arcade/labels/valid'
img_dir = '/data/Arcade_dataset/arcade/images/valid'
img_dir2 = '/data/yolov5/00_dataset_plot/small_valid'
save_dir = img_dir2 + '_2'

def plot_panoid(pano_id, img):
    cv2.rectangle(img, (0, 5), (420, 30), (137, 13, 58), thickness=-1)
    cv2.putText(img, 'PanoID:' + pano_id, (2, 25), 0, 0.8, (255, 255, 255), 2)

def read_label(label):
    table = pd.read_csv(os.path.join(label_dir, label),
                            index_col=False,
                            header=None,
                            sep=' ',
                            dtype =float).iloc[:, 1:].sort_values(by=[1])
    table = torch.tensor(np.array(table))
    table = xywh2xyxy(table) * 640
    return table

imgs = os.listdir(img_dir)
labels = os.listdir(label_dir)
os.makedirs(save_dir)

for i in tqdm(imgs):
    if i.replace('.jpg', '.txt') in labels:
        pano_id = i.split('$')[0]
        xyxys = read_label(i.replace('.jpg', '.txt')).int()
        img = cv2.imread(os.path.join(img_dir, i))
        for xyxy in xyxys:
            plot_one_box(x=xyxy, im=img, color=(0, 0, 200), label=None, line_thickness=3)
        # plot_panoid(pano_id, img)
        cv2.imwrite(os.path.join(save_dir, i), img)

# for i in tqdm(labels):
#     if i.replace('.txt', '.jpg') not in imgs:
#         pano_id = i.split('$')[0]
#         xyxys = read_label(i).int()
#         img = cv2.imread(os.path.join(img_dir, i))
#         for xyxy in xyxys:
#             plot_one_box(x=xyxy, im=img, color=(0, 0, 200), label=None, line_thickness=3)
#         # plot_panoid(pano_id, img)
#         cv2.imwrite(os.path.join(save_dir, i), img)

            




            
        


    




