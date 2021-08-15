import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import data_helper
import torchvision
from matplotlib import pyplot as plt
import imageio
import pfm


class StereoPair(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.stereo_image_folders = get_image_pair_names(self.data_path)
        self.mode = None
        self.transform = None

    def __len__(self):
        return len(self.stereo_image_folders)

    def __getitem__(self, idx):
        image_pair_path = self.stereo_image_folders[idx]
        left_image = torchvision.io.read_image(image_pair_path[0])
        right_image = torchvision.io.read_image(image_pair_path[1])
        image_index = image_pair_path[0].split("/")[-1][:-4]
        disparity_path = os.path.join(image_pair_path[0].split("/RGB")[0], "disparity", image_index + ".pfm")
        disparity,scale = data_helper.readPFM(disparity_path)
        return (left_image, right_image), disparity


def get_image_pair_names(data_path):
    image_pair_list = []
    for sub_folder in os.listdir(data_path):
        data_path_1 = os.path.join(data_path, sub_folder, "RGB_cleanpass")
        for pair_type in os.listdir(data_path_1):
            data_path_2 = os.path.join(data_path_1, pair_type)
            image_pair = []
            for image_name in os.listdir(data_path_2):
                image_pair.append(os.path.join(data_path_2, image_name))
            image_pair_list.append(image_pair)
    return image_pair_list


