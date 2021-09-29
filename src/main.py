# from dataloader.sceneflow import list_file_path
from dataloader.kitti import list_file_path
from dataloader import stereo_loader
from dataloader import transforms
from torch.utils.data import DataLoader
from models import stereo_depth
from matplotlib import pyplot as plt
import configparser
import torch
from tqdm import tqdm
import numpy as np
from torch import nn

# read config files
config = configparser.ConfigParser()
config.read("configs/kitti.config")
learning_rate = config.getfloat("Training", "learning_rate")
epochs = config.getint("Training", "epochs")
eval_freq = config.getint("Training", "eval_freq")
save_freq = config.getint("Training", "save_freq")
batch_size = config.getint("Training", "batch_size")
datapath_training = config.get("Data", "datapath_training")

# check if CUDA exists
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#read training and eval files
stereo_path_list_train = list_file_path.get_image_pair_names(datapath_training, mode="train")
stereo_path_list_val = list_file_path.get_image_pair_names(datapath_training, mode="eval")

# make dataset objects for training and eval
stereo_dataset_train = stereo_loader.StereoPair(stereo_path_list_train,
                                                transforms=transforms.get_transforms(), mode="train")
stereo_dataset_val = stereo_loader.StereoPair(stereo_path_list_val,
                                              transforms=transforms.get_transforms(), mode="eval")
# make dataloader objects for training and eval
dataloader_train = DataLoader(stereo_dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
dataloader_validation = DataLoader(stereo_dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)


# save model function
def save_model(model, stats, model_name):
    model_dict = {"model": model, "stats": stats}
    torch.save(model_dict, "../models/" + model_name + ".pth")


def train(m=None):
    # if m is none start from scratch else continue training
    if m is None:
        model = stereo_depth.StereoDepth()
        model = nn.DataParallel(model).to(device)
        init_epoch = 0
        stats = {
            "epoch": [],
            "train_loss": [],
            "valid_loss": [],
            "accuracy": []
        }
    else:
        model = m["model"]
        stats = m["stats"]
        init_epoch = list(stats.keys())[-1]
    criterion = torch.nn.SmoothL1Loss().to(device) #smooth L1 Loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate) # adam optimizer
    torch.autograd.set_detect_anomaly(True)

    # start training
    loss_hist = []
    for epoch in range(init_epoch, epochs):
        loss_list = []
        progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
        for i, batch in progress_bar:
            left_image = batch[0].to(device) # read left image
            right_image = batch[1].to(device) # read right image
            gt_disparity = batch[2].to(device) # read disaprity ( output )

            optimizer.zero_grad()
            predicted_disparity = model(left_image, right_image) # send left and right image through model

            non_zero_disp_indices = torch.where(gt_disparity > 0) # only find loss where disparity > 0

            loss = criterion(predicted_disparity[non_zero_disp_indices], gt_disparity[non_zero_disp_indices]) # find loss
            loss_list.append(loss.item()) # append to loss list
            # backpropogate
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch {0 + epoch} Iter {i + 1}: loss {loss.item():.5f}. ")

        print(np.mean(loss_list)) # find mean loss
        loss_hist.append(np.mean(loss_list))

        # store states
        stats[epoch] = epoch
        stats['train_loss'].append(loss_hist[-1])

        if epoch % eval_freq == 0:
            three_pe, valid_loss = evaluate(model,plot=False)
            print(f"3PE at epoch {epoch}: {round(three_pe * 100.0, 3)}%")
        else:
            three_pe, valid_loss = -1, -1
        stats["accuracy"].append(three_pe)
        stats["valid_loss"].append(valid_loss)

        # saving checkpoint
        if epoch % save_freq == 0:
            save_model(model, stats, "model_trilinear_aug")


@torch.no_grad()
def evaluate(model,plot=False):
    criterion = torch.nn.SmoothL1Loss()
    three_pe_list = []
    loss_list = []

    for batch in dataloader_validation:
        left_image = batch[0].to(device)
        right_image = batch[1].to(device)
        gt_disparity = batch[2].squeeze(1).to(device)
        l = batch[3]
        r = batch[4]
        non_zero_disp_indices = torch.where(gt_disparity > 0) # only pixels with non zero disparity for 3PE
        zero_disp_indices = torch.where(gt_disparity == 0)
        predicted_disparity = model(left_image, right_image)
        predicted_disparity[zero_disp_indices] = 0
        if plot:
            fig = plt.figure(figs=(20,20))
            cols = 1
            rows = 3
            fig.add_subplot(rows, cols, 1)
            plt.imshow(l.squeeze(0).permute(1,2,0))
            fig.add_subplot(rows, cols, 2)
            plt.imshow(r.squeeze(0).permute(1,2,0))
            fig.add_subplot(rows, cols, 3)
            plt.imshow(predicted_disparity.cpu().squeeze(0))
            plt.show()
            plt.pause(0.5)
            plt.close()
        three_pe = three_pixel_error(gt_disparity[non_zero_disp_indices], predicted_disparity[non_zero_disp_indices])
        three_pe_list.append(three_pe.to("cpu"))

        loss = criterion(predicted_disparity[non_zero_disp_indices], gt_disparity[non_zero_disp_indices])
        loss_list.append(loss.item())

    return np.mean(three_pe_list), np.mean(loss_list)


def three_pixel_error(gt_disparity, predicted_disparity):
    abs_diff = torch.abs(gt_disparity - predicted_disparity) # find absolute differeence in target and predicted
    correct_disparity_pixels = torch.where((abs_diff < 3) | (abs_diff < 0.05 * gt_disparity)) # apply 3PE condition
    correct_disparity = torch.zeros_like(gt_disparity)  # empty tensor
    correct_disparity[correct_disparity_pixels] = 1 # make correct prediction locations as 1
    three_pixel_error = 1 - correct_disparity.mean() # take mean and subtract from 1 to get error
    return three_pixel_error


# for training the model
m = None
train(m)

# for reading and evaluating a model
m = torch.load("../models/model_trilinear_aug.pth")
print(evaluate(m['model']))