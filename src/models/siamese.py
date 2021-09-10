from torch import nn
import configparser

config = configparser.ConfigParser()
config.read("configs/kitti.config")

in_channels = config.getint("Siamese", "in_channels")
channels = config.getint("Siamese", "channels")
channels2d = [int(c) for c in config.get("Siamese", "channels2d")[1:-1].split(",")]
stride2d = [int(s) for s in config.get("Siamese", "stride2d")[1:-1].split(",")]
kernel_size_res = config.getint("Siamese", "kernel_size_res")
kernel_size_siamese = config.getint("Siamese", "kernel_size_siamese")
num_res_blocks = config.getint("Siamese", "num_res_blocks")

class Siamese(nn.Module):
    def __init__(self, block):
        super(Siamese, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.block = block
        self.num_blocks = num_res_blocks
        self.conv1 = nn.Conv2d(self.in_channels, self.channels, kernel_size_siamese, stride=1, padding=3) # keeps same dim, increases channels
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU()
        layers = list()
        # layers.append(self.block(in_channels, channels2d[0], kernel_size_res, stride=2, padding=1, change_dim=True))
        for i in range(0, num_res_blocks):
            layers.append(self.block(channels2d[i], channels2d[i+1], kernel_size_res, stride=stride2d[i], padding=1, change_dim=True))
        self.residual_sequence = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.residual_sequence(x)
        return x
