import cv2
import os
import shutil
import argparse
from tqdm import tqdm

from numpy.core.fromnumeric import var

from detect import run as detection
from utils.general import check_requirements, colorstr


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='artifacts/coco_small/0_0.3876.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='now_url.txt', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3876, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_false', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    return opt

def get_net_img(url):
    img = None
    cap = cv2.VideoCapture(url)
    file_name = url.split('/')[-1]
    if cap.isOpened():
        _, img=cap.read()
    cv2.imwrite(f'__webcache__/{file_name}', img)


def main(opt):
    opt = vars(opt)
    opt['nosave'] = True
    opt['exist_ok'] = False
    opt['save_conf'] = False
    urls = open(opt['source'], 'r')
    urls = list(i.replace('\n', '') for i in urls)
    opt['source'] = '__webcache__'
    cnt = 0
    print('Downloading WebImages!')
    errors = []
    for url in tqdm(urls):
        cnt += 1
        try:
            get_net_img(url)
        except:
            errors.append(url)
        if cnt % 1000 == 0 or cnt == len(urls):
            print(f'Batch {cnt // 1000} download success!')
            print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in opt.items()))
            check_requirements(exclude=('tensorboard', 'thop'))
            detection(**opt)
            shutil.rmtree('__webcache__')
            os.makedirs('__webcache__')
    with open('error.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(errors))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)


