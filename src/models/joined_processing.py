import torch
from torch import nn
import configparser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = configparser.ConfigParser()
config.read("configs/sceneflow.config")

in_channels = config.getint("Siamese", "channels")

height = config.getint("Data", "height")
width = config.getint("Data", "width")
max_disparity = config.getint("Data", "max_disparity")

channels3d = [int(c) for c in config.get("CostVolume", "channels3d")[1:-1].split(",")]
num_cost_blocks = config.getint("CostVolume", "num_cost_blocks")

batch_size = config.getint("Training", "batch_size")



class CostVolume(nn.Module):
    def __init__(self, block3d):
        super(CostVolume, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.max_disparity = max_disparity
        self.in_channels = in_channels
        self.channels = channels3d
        self.num_cost_blocks = num_cost_blocks
        layers = list()
        for i in range(self.num_cost_blocks):
            layers.append(block3d(self.channels[i], self.channels[i + 1], kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                                  change_dim=True))
        self.process_volume = nn.Sequential(*layers)

    def forward(self, x):
        left_feat, right_feat = x
        cost = self.features_to_cost_volume(left_feat, right_feat)
        processed_volume = self.process_volume(cost)
        return processed_volume

    def features_to_cost_volume(self, left_feat, right_feat):

        cost = torch.Tensor(self.batch_size, self.in_channels * 2, self.max_disparity // 4, self.height // 4,
                            self.width // 4).to(device)
        for i in range(self.max_disparity // 4):
            if i == 0:
                cost[:, :self.in_channels, i, :, :] = left_feat
                cost[:, self.in_channels:, i, :, :] = right_feat
            else:
                cost[:, :self.in_channels, i, :, i:] = left_feat[:, :, :, i:]
                cost[:, self.in_channels:, i, :, i:] = right_feat[:, :, :, :-i]

        return cost
