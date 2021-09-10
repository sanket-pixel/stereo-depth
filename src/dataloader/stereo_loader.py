import os
import torch
from torch.utils.data import Dataset
import torchvision
import cv2
import numpy as np
from PIL import Image
from . import dataset_utils
import configparser
from torchvision.io.image import ImageReadMode

config = configparser.ConfigParser()
config.read(os.path.join("configs", "kitti.config"))


class StereoPair(Dataset):

    def __init__(self, stereo_paths, transforms, mode):
        self.stereo_paths = stereo_paths
        self.mode = mode
        self.transforms = transforms
        self.H = config.getint("Data", "H")
        self.W = config.getint("Data", "W")
        self.h = config.getint("Data", "height")
        self.w = config.getint("Data", "width")

    def __len__(self):
        return len(self.stereo_paths)

    def __getitem__(self, idx):
        image_pair_path = self.stereo_paths[idx]
        left_image = torchvision.io.read_image(image_pair_path[0]).float() / 255.0
        l= left_image
        right_image = torchvision.io.read_image(image_pair_path[1]).float() / 255.0
        r = right_image
        disparity = torch.from_numpy(np.array(Image.open(image_pair_path[2])) / 256).float().unsqueeze(0)
        # disparity = dataset_utils.readPFM(image_pair_path[2])[0].unsqueeze(0)
        if self.mode == "train":
            rand_h = torch.randint(0, self.H - self.h, (1,))
            rand_w = torch.randint(0, self.W - self.w, (1,))
            left_image = self.crop_image(left_image, rand_h, rand_w)
            right_image = self.crop_image(right_image, rand_h, rand_w)
            disparity = self.crop_image(disparity, rand_h, rand_w).squeeze(0)
        left_image = self.transforms(left_image)
        right_image = self.transforms(right_image)
        if self.mode=="eval":
            return left_image,right_image,disparity,l,r
        return left_image, right_image, disparity

    def crop_image(self, image, rand_h, rand_w):
        return image[:, rand_h:rand_h + self.h, rand_w:rand_w + self.w]
