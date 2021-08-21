from torch import nn


class ResidualBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, change_dim=None):
        super(ResidualBlock3D, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.change_dim = change_dim

        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.change_dim:
            identity = self.bn1(self.conv1x1x1(identity))

        x += identity
        x = self.relu(x)
        return x
