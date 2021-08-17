import torch
from dataloader import list_file_path
from dataloader import StereoLoader
from dataloader import transforms
from models import stereo_depth, siamese, residual_block
import configparser

config = configparser.ConfigParser()
config.read("configs/sceneflow.config")

datapath = config.get("Data","datapath")
stereo_path_list = list_file_path.get_image_pair_names(datapath)
stereo_dataset = StereoLoader.StereoPair(stereo_path_list,transforms=transforms.get_transforms())

in_channels = int(config.get("Siamese","in_channels"))
channels = int(config.get("Siamese","channels"))
kernel_size_res = int(config.get("Siamese","kernel_size_res"))
kernel_size_siamese = int(config.get("Siamese","kernel_size_siamese"))
num_res_blocks = int(config.get("Siamese","num_res_blocks"))

block = residual_block.ResidualBlock
siamese = siamese.Siamese(block,in_channels,channels,kernel_size_res,kernel_size_siamese,num_res_blocks)
model = stereo_depth.StereoDepth(siamese)

stereo_pair = stereo_dataset[0]
l,r = model(stereo_pair)