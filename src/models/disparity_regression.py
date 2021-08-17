import torch
from torch import nn

class SoftRegression(nn.Module):
    def __init__(self,scale_factor,mode='bilinear', max_disparity=192):
        super(SoftRegression, self).__init__()
        self.upsample = nn.Upsample(scale_factor=4,mode=mode)
        self.channelwise_softmax = nn.Softmax(dim=1)
        self.disparity_values = torch.arange(max_disparity).float()


    def forward(self,x):
        x = self.upsample(x)
        x = x.squeeze(1)
        x = self.channelwise_softmax(x)
        x = torch.tensordot(x,self.disparity_values.float(),[[1],[0]])
        return x

