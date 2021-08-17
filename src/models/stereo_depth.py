from torch import nn
import torch

class StereoDepth(nn.Module):
    def __init__(self, siamese, cost_volume, channels, batch_size, height, width, max_disparity):
        super(StereoDepth, self).__init__()
        self.siamese = siamese
        self.cost_volume = cost_volume




    def forward(self,x):
        left_image = x[0].unsqueeze(0)
        right_image = x[1].unsqueeze(0)
        disparity = x[2]
        left_feature = self.siamese(left_image)
        right_feature = self.siamese(right_image)
        cost = self.cost_volume((left_feature,right_feature))
        return cost