import torch
from dataloader import list_file_path
from dataloader import stereo_loader
from dataloader import transforms
from models import stereo_depth, siamese, residual_block, joined_processing, residual_block_3d
import configparser

config = configparser.ConfigParser()
config.read("configs/sceneflow.config")

datapath = config.get("Data","datapath")
stereo_path_list = list_file_path.get_image_pair_names(datapath)
stereo_dataset = stereo_loader.StereoPair(stereo_path_list,
                                          transforms=transforms.get_transforms())

in_channels = config.getint("Siamese","in_channels")
channels = config.getint("Siamese","channels")
kernel_size_res = config.getint("Siamese","kernel_size_res")
kernel_size_siamese = config.getint("Siamese","kernel_size_siamese")
num_res_blocks = config.getint("Siamese","num_res_blocks")
height = config.getint("Siamese","height")
width = config.getint("Siamese","width")

num_cost_blocks = config.getint("CostVolume","num_cost_blocks")
max_disparity = config.getint("CostVolume","max_disparity")
channels3d = [int(c) for c in config.get("CostVolume", "channels3d")[1:-1].split(",")]

batch_size = config.getint("StereoDepth","batch_size")

block = residual_block.ResidualBlock
block3d = residual_block_3d.ResidualBlock3D
siamese = siamese.Siamese(block,in_channels,channels,kernel_size_res,
                          kernel_size_siamese,num_res_blocks)
cost_volume = joined_processing.CostVolume(block3d=block3d,num_cost_blocks=num_cost_blocks, batch_size=1,
                                           in_channels=channels,channels=channels3d, height=height, width=width,
                                           max_disparity=max_disparity)
model = stereo_depth.StereoDepth(siamese, cost_volume, channels=channels, batch_size=batch_size,
                                 height=height, width=width, max_disparity=max_disparity)

stereo_pair = stereo_dataset[0]
cost = model(stereo_pair)
