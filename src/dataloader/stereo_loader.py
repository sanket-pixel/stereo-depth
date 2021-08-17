import os
import torch
from torch.utils.data import Dataset
import torchvision
from matplotlib import pyplot as plt

from . import dataset_utils



class StereoPair(Dataset):

    def __init__(self, stereo_paths, transforms):
        self.stereo_paths = stereo_paths
        self.mode = None
        self.transforms = transforms

    def __len__(self):
        return len(self.stereo_paths)

    def __getitem__(self, idx):
        image_pair_path = self.stereo_paths[idx]
        left_image = torchvision.io.read_image(image_pair_path[0]).float()/255.0
        right_image = torchvision.io.read_image(image_pair_path[1]).float()/255.0
        disparity,scale = dataset_utils.readPFM(image_pair_path[2])
        left_image = self.transforms(left_image)
        right_image = self.transforms(right_image)
        return left_image, right_image, disparity



