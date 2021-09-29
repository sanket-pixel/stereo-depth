import os
import torch
from torch.utils.data import Dataset
import torchvision
import cv2
import numpy as np
from PIL import Image
from . import  transforms
from . import dataset_utils
import configparser
from torchvision.io.image import ImageReadMode


config = configparser.ConfigParser()
config.read(os.path.join("configs", "kitti.config"))


class StereoPair(Dataset):

    def __init__(self, stereo_paths, transforms, mode):
        self.stereo_paths = stereo_paths
        self.mode = mode
        # get image dimensions
        self.H = config.getint("Data", "H")
        self.W = config.getint("Data", "W")
        self.h = config.getint("Data", "height")
        self.w = config.getint("Data", "width")

    def __len__(self):
        return len(self.stereo_paths)

    def __getitem__(self, idx):
        image_pair_path = self.stereo_paths[idx] # read image index
        color_changes, normalize = transforms.get_transforms() # get augmentations and normalization transform functions
        left_image = torchvision.io.read_image(image_pair_path[0]).float() / 255.0 # read left image
        l = left_image
        right_image = torchvision.io.read_image(image_pair_path[1]).float() / 255.0 # read right image
        r = right_image
        disparity = torch.from_numpy(np.array(Image.open(image_pair_path[2])) / 256).float().unsqueeze(0) # read disparity
        # if training mode then crop and augment
        if self.mode == "train":
            # get random crop for every pair
            rand_h = torch.randint(0, self.H - self.h, (1,))
            rand_w = torch.randint(0, self.W - self.w, (1,))
            left_image = self.crop_image(left_image, rand_h, rand_w)
            right_image = self.crop_image(right_image, rand_h, rand_w)
            disparity = self.crop_image(disparity, rand_h, rand_w).squeeze(0) # crop disparity as well
            left_image = color_changes(left_image) # apply augmentation to left
            right_image = color_changes(right_image) # apply augmentation to right
        left_image = normalize(left_image)
        right_image = normalize(right_image)
        if self.mode=="eval":
            return left_image,right_image,disparity,l,r
        return left_image, right_image, disparity

    def crop_image(self, image, rand_h, rand_w):
        return image[:, rand_h:rand_h + self.h, rand_w:rand_w + self.w]
