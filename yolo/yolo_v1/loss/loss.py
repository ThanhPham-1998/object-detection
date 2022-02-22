import torch

class Loss:
    def __init__(self, ) -> None:
        pass




    # def IOU(self, boxA, boxB):
    #     inter_x = max(0, min(boxA[0] + boxA[2] * 0.5, boxB[0] + boxB[2] * 0.5) - 
    #                         max(boxA[0] - boxA[2] * 0.5, boxB[0] - boxB[2] * 0.5))

    #     inter_y = max(0, min(boxA[1] + boxA[3] * 0.5, boxB[1] + boxB[3] * 0.5) - 
    #                         max(boxA[1] - boxA[3] * 0.5, boxB[1] - boxB[3] * 0.5))
        
    #     inter = inter_x * inter_y
    #     union = boxA[2] * boxA[3] + boxB[2] * boxB[3] - inter
        
    #     return inter / union
    
    def IOU(self, boxA, boxB):
        inter_x = torch.min(boxA[:, :, :, 0] + boxA[:, :, :, 2] * 0.5, boxB[:, :, :, 0] + boxB[:, :, :, 2] * 0.5) - \
                  torch.max(boxA[:, :, :, 0] - boxA[:, :, :, 2] * 0.5, boxB[:, :, :, 0] - boxB[:, :, :, 2] * 0.5)
        inter_y = torch.min(boxA[:, :, :, 1] + boxA[:, :, :, 3] * 0.5, boxB[:, :, :, 1] + boxB[:, :, :, 3] * 0.5) - \
                  torch.max(boxA[:, :, :, 1] - boxA[:, :, :, 3] * 0.5, boxB[:, :, :, 1] - boxB[:, :, :, 3] * 0.5)
        inter = torch.max(0, inter_x) * torch.max(0, inter_y)
        union = boxA[:, :, :, 2] * boxA[:, :, :, 3] + boxB[:, :, :, 2] * boxB[:, :, :, 3] - inter
        return inter / union
    
     