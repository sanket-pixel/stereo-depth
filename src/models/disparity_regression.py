import torch
from torch import nn

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
        self.upsample = nn.Upsample(scale_factor=self.scale_factor, mode=self.mode)
        self.channelwise_softmax = nn.Softmax(dim=1)
        self.disparity_values = torch.arange(self.max_disparity).float().to(device)

    def forward(self, x):
        x = self.upsample(x)
        x = x.squeeze(1)
        x = self.channelwise_softmax(x)
        x = torch.tensordot(x, self.disparity_values.float(), [[1], [0]])
        return x
