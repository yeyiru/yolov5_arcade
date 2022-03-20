from detect import main as detect_main
from val import run as val_run
import os
# 採用在Val中map.5.95前4名

import argparse

for mode1 in ['nopre', 'coco', 'beauty']:
    for mode2 in ['small', 'merge']:
        mode = mode1 + '_' + mode2
        for pt in os.listdir(os.path.join('./artifacts', mode)):
            tmp = pt.split('_')
            iou_thres = float(tmp[-1].replace('.pt', ''))
            idx = tmp[0]
            if mode1 == 'coco' and mode2 == 'small' and idx == '2':
                opt = dict(
                    agnostic_nms=False,
                    augment=False,
                    classes=None,
                    conf_thres=0.647,
                    device='0',
                    exist_ok=True,
                    half=False,
                    hide_conf=False,
                    hide_labels=False,
                    imgsz=640,
                    iou_thres=iou_thres,
                    line_thickness=3,
                    max_det=1000,
                    name='_'.join([mode, idx]),
                    nosave=False,
                    project=os.path.join('runs_val'),
                    save_conf=True,
                    save_crop=False,
                    save_txt=True,
                    source='/data/Arcade_dataset/arcade/images/test/',
                    update=False,
                    view_img=False,
                    visualize=False,
                    weights=os.path.join('./artifacts', mode, pt))

                opt = argparse.Namespace(**opt)
                detect_main(opt)