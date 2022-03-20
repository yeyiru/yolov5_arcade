import torch

def w_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])
    
    area1 = box_area(box1.T)[:, None]
    w1 = box1.T[2] - box1.T[0]
    tm = w1[:, None]
    w2 = box2.T[2] - box2.T[0]
    # area2 = box_area(box2.T)

    # # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # tmp1 = box1[:, None, 2:]
    # tmp2 = box1[:, 2:]
    # tmp3 = box2[:, 2:]
    # min = torch.min(box1[:, None, 2:], box2[:, 2:])
    # max = torch.max(box1[:, None, :2], box2[:, :2])
    # s = min - max
    tmp = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0)
    pr = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # if tmp = 
    w_inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0)[:, :, 0]
    return w_inter / (w1[:, None] + w2 - w_inter)  # iou = inter / (area1 + area2 - inter)