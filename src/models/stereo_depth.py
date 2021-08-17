from torch import nn


class StereoDepth(nn.Module):
    def __init__(self, siamese):
        super(StereoDepth, self).__init__()
        self.siamese = siamese

    def forward(self,x):
        left_image = x[0].unsqueeze(0)
        right_image = x[1].unsqueeze(0)
        disparity = x[2]
        left_feature = self.siamese(left_image)
        right_feature = self.siamese(right_image)
        return left_feature, right_feature