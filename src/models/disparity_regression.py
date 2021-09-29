import torch
from torch import nn
from matplotlib import pyplot as plt
import configparser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("configs/kitti.config")

scale_factor = config.getint("DisparityRegression", "scale_factor")
mode = config.get("DisparityRegression", "mode")
max_disparity = config.getint("Data", "max_disparity")


class SoftRegression(nn.Module):
    def __init__(self):
        super(SoftRegression, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.max_disparity = max_disparity
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode) # upsampling to match image size
        self.channelwise_softmax = nn.Softmax(dim=1) # apply channelwise softmax
        self.disparity_values = torch.arange(self.max_disparity).float().to(device) # define disparity values

    def forward(self, x):
        x = self.upsample(x) # upsample
        x = x.squeeze(1)
        x = self.channelwise_softmax(x) # channelwise softmax
        x = torch.tensordot(x, self.disparity_values.float(), [[1], [0]]) #take weighted average for soft regression
        return x
