from utils import IOU
import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, S=7, B=2, C=3, lambda_noobj=0.5, lambda_coord=5):
        super().__init__()
        self.mse = nn.MSELoss(reduce='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord

    def forward(self, preds, target):

        """
            preds: [c1, c2, ..., cn, x1, y1, x2, y2, conf1, x3, y3, x4, y4, conf2]
        """
        iou_b1 = IOU(preds[..., self.C + 1: self.C + 5], target[..., self.C + 1:self.C + 5])
        iou_b2 = IOU(preds[..., self.C + 6: self.C + 10], target[..., self.C + 6:self.C + 10])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dims=0)

