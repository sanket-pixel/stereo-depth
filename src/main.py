from dataloader import list_file_path
from dataloader import stereo_loader
from dataloader import transforms
from torch.utils.data import DataLoader
from models import stereo_depth
import configparser
import torch
from tqdm import tqdm
import numpy as np

config = configparser.ConfigParser()
config.read("configs/sceneflow.config")
learning_rate = config.getfloat("Training", "learning_rate")
epochs = config.getint("Training", "epochs")
eval_freq = config.getint("Training", "eval_freq")
save_freq = config.getint("Training", "save_freq")
batch_size = config.getint("Training", "batch_size")
datapath = config.get("Data", "datapath")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stereo_path_list = list_file_path.get_image_pair_names(datapath)
stereo_dataset = stereo_loader.StereoPair(stereo_path_list,
                                          transforms=transforms.get_transforms())

dataloader_train = DataLoader(stereo_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
dataloader_validation = DataLoader(stereo_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


# save model
def save_model(model, stats, model_name):
    model_dict = {"model": model, "stats": stats}
    torch.save(model_dict, "../models/" + model_name + ".pth")


def train():
    model = stereo_depth.StereoDepth()
    model = model.to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    torch.autograd.set_detect_anomaly(True)
    stats = {
        "epoch": [],
        "train_loss": [],
        "valid_loss": [],
        "accuracy": []
    }
    init_epoch = 0
    loss_hist = []
    for epoch in range(init_epoch, epochs):
        loss_list = []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, batch in progress_bar:
            left_image = batch[0].to(device)
            right_image = batch[1].to(device)
            gt_disparity = batch[2].to(device)

            optimizer.zero_grad()
            predicted_disparity = model(left_image, right_image)

            loss = criterion(predicted_disparity, gt_disparity)
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {0 + 1} Iter {i + 1}: loss {loss.item():.5f}. ")

        loss_hist.append(np.mean(loss_list))
        stats[epoch]=epoch
        stats['train_loss'].append(loss_hist[-1])

# stereo_pair = stereo_dataset[0]
# predicted_disparity, gt_disparity = model(stereo_pair)

train()