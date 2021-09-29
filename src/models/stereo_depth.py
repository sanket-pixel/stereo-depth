from torch import nn
import torch
import configparser
from . import siamese, joined_processing, disparity_regression, residual_block, residual_block_3d

config = configparser.ConfigParser()
config.read("configs/kitti.config")


class StereoDepth(nn.Module):
    def __init__(self):
        super(StereoDepth, self).__init__()
        self.block = residual_block.ResidualBlock
        self.block3d = residual_block_3d.ResidualBlock3D
        self.siamese = siamese.Siamese(block=self.block)
        self.cost_volume = joined_processing.CostVolume(block3d=self.block3d)
        self.disparity_regression = disparity_regression.SoftRegression()

    def forward(self, left_image, right_image):
        left_feature = self.siamese(left_image) # process left image through siamese
        right_feature = self.siamese(right_image) #process right image through siamese
        cost = self.cost_volume((left_feature, right_feature)) # concat them into a 3D cost volume and process it
        predicted_disparity = self.disparity_regression(cost) # apply soft regression
        return predicted_disparity
