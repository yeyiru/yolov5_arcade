from torch.cuda import device
import wandb
import argparse
from val import run

hyper_default = dict(
    data = '/data/yolov5/data/coco.yaml',
    weights = '/data/yolov5/yolov5x.pt',
    device = '3',
    project='runs/coco',
    conf_thres = 0.001,
    iou_thres=0.65,
    iou_mode = 'box_iou', #w_iou box_iou
    test_iou_thres = 0.65
)

wandb.init(project="Yolo_test", config=hyper_default)
config = wandb.config

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=config['data'], help='dataset.yaml path')
parser.add_argument('--weights', nargs='+', type=str, default=config['weights'], help='model.pt path(s)')
parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')

parser.add_argument('--conf_thres', type=float, default=config['conf_thres'], help='confidence threshold')
parser.add_argument('--iou_thres', type=float, default=config['iou_thres'], help='NMS IoU threshold')

parser.add_argument('--task', default='val', help='train, val, test, speed or study')
parser.add_argument('--device', default=config['device'], help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--verbose', action='store_false', help='report mAP by class')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
parser.add_argument('--project', default=config['project'], help='save to project/name')
parser.add_argument('--name', default='exp', help='save to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--iou_mode', default=config['iou_mode'], help='w_iou or box_iou ')
parser.add_argument('--test_iou_thres', default=config['test_iou_thres'], help='Test IoU threshold')
opt = parser.parse_args()


tmp, maps, t = run(**vars(opt))

mp = tmp[0]
mr = tmp[1]
map50 = tmp[2]
map = tmp[3]
f1 = tmp[-1]

metrics = {
    'P': mp,
    'R': mr,
    'mAP@0.5': map50,
    'mAP@0.5:0.95': map,
    'F1': f1
}
wandb.log(metrics)