from torch import nn

class ResidualBlock(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, change_dim=None):
        super(ResidualBlock, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.change_dim = change_dim

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv1x1 = nn.Conv2d(channels,channels,kernel_size=1,stride=2,padding=0)


    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if self.change_dim:
            identity = self.bn1(self.conv1x1(identity))
            identity = self.bn2(self.conv1x1(identity))
        x += identity
        x = self.relu(x)

        return x


