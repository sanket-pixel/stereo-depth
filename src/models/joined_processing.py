import torch
from torch import nn

class CostVolume(nn.Module):
    def __init__(self, block3d, num_cost_blocks, batch_size, in_channels, channels, height, width, max_disparity):
        super(CostVolume, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.max_disparity = max_disparity
        self.in_channels = in_channels
        self.channels = channels
        self.num_cost_blocks = num_cost_blocks
        layers = list()
        for i in range(self.num_cost_blocks):
            layers.append(block3d(channels[i], channels[i+1], kernel_size=(1,3,3), stride=1, padding=(0,1,1), change_dim=True))
        self.process_volume = nn.Sequential(*layers)

    def forward(self, x):
        left_feat, right_feat = x
        cost = self.features_to_cost_volume(left_feat, right_feat)
        processed_volume = self.process_volume(cost)
        return processed_volume

    def features_to_cost_volume(self, left_feat, right_feat):

        cost = torch.Tensor(self.batch_size,self.in_channels * 2, self.max_disparity // 4, self.height // 4,
                            self.width // 4)
        for i in range(self.max_disparity // 4):
            if i == 0:
                cost[:, :self.in_channels, i, :, :] = left_feat
                cost[:, self.in_channels:, i, :, :] = right_feat
            else:
                cost[:, :self.in_channels, i, :, i:] = left_feat[:, :, :, i:]
                cost[:, self.in_channels:, i, :, i:] = right_feat[:, :, :, :-i]

        return cost
