import torch
from collections import Counter

def IOU(boxes_preds, boxes_labels, box_format='midpoint'):
    
    """
        boxes_preds (tensor): [batch_size, 4],
        boxes_labels (tensor): [batch_size, 4],
        box_format (str): 
            + midpoint: (x, y, w, h)
            + corners: (x1, y1, x2, y2)
    """
    if box_format == 'midpoints':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] * 0.5
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] * 0.5
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] * 0.5
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] * 0.5

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] * 0.5
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] * 0.5
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] * 0.5
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] * 0.5

    if box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    area1 = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    area2 = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = area1 + area2 - inter + 1e-6

    return inter / union



def non_max_suppression(bboxes, iou_threshold, threshold, box_format='corners'):
    """
    - bboxes: List of bbox [cls_pred, prob_score, x1, y1, x2, y2]
    - iou_threshold: 
    - threshold: threshold to remove predicted bboxes (independent of IOU)
    - box_format: 
    """

    assert type(bboxes) == list, f'bboxes must is a list, but is found {type(bboxes)}'

    # get bbox, which have prob_score >= threshold
    bboxes = [bbox for bbox in bboxes if bbox[1] >= threshold]
    bboxes = sorted(bboxes, key=lambda bbox: bbox[1], reverse=True)
    
    bboxes_after_nns = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or IOU(torch.tensor(chosen_box[2:]), torch.tensor(box[:2], box_format)) < iou_threshold
        ]
        bboxes_after_nns.append(chosen_box)
    
    return bboxes_after_nns


def mAP(pred_bboxes, true_bboxes, iou_threshold=0.5, box_format='midpoint', num_classes=2):
    """
        - pred_bboxes: List of all bbox: [train_idx, cls_pred, prob_score, x1, y1, x2, y2]
        - true_bboxes: 
        - iou_threshold:
        - box_format:
        - num_classes:

    """
    average_precisions = []
    eps = 1e-6
    for c in range(num_classes):
        detections = [pred for pred in pred_bboxes if pred[1] == c].sort(key=lambda pred: pred[2], reverse=True)
        gts = [gt for gt in true_bboxes if gt[1] == c]
        amount_bboxes = Counter([gt[0] for gt in gts])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(gts)

        if total_true_bboxes == 0:
            continue
        for detection_idx, detection in enumerate(detections):
            gt_img = [bbox for bbox in gts if bbox[0] == detection[0]]
            # num_gts = len(gt_img)
            best_iou = 0
            for idx, gt in enumerate(gt_img):
                iou = IOU(torch.tensor(detection[3:], torch.tensor(gt[3:], box_format=box_format)))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + eps)
        precisions = torch.device(TP_cumsum, (TP_cumsum + FP_cumsum + eps))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)