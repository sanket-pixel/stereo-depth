from dataloader import list_file_path
from dataloader import stereo_loader
from dataloader import transforms
from models import stereo_depth, siamese, residual_block, joined_processing, residual_block_3d, disparity_regression
import configparser
import torch

config = configparser.ConfigParser()
config.read("configs/sceneflow.config")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

datapath = config.get("Data", "datapath")
stereo_path_list = list_file_path.get_image_pair_names(datapath)
stereo_dataset = stereo_loader.StereoPair(stereo_path_list,
                                          transforms=transforms.get_transforms())

learning_rate = config.getfloat("Training", "learning_rate")
epochs = config.getint("Training", "epochs")
eval_freq = config.getint("Training", "eval_freq")
save_freq = config.getint("Training", "save_freq")


model = stereo_depth.StereoDepth()


def train():
    pass

# stereo_pair = stereo_dataset[0]
# predicted_disparity, gt_disparity = model(stereo_pair)
