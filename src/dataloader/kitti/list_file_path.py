import os
import configparser

config = configparser.ConfigParser()
config.read(os.path.join("configs", "kitti.config"))


def get_image_pair_names(data_path, mode="train"):
    image_pair_list = []
    left_image_folder = "image_2"
    right_image_folder = "image_3"
    disparity_folder = "disp_occ_0"
    split_file = open(config.get("Data","split_path_"+mode))
    image_name_list = split_file.readlines()
    image_name_list = [i.replace("\n", "") for i in image_name_list]
    for img in image_name_list:
        left_image_path = os.path.join(data_path, left_image_folder, img + "_10.png")
        right_image_path = os.path.join(data_path, right_image_folder, img + "_10.png")
        disp_occ = os.path.join(data_path, disparity_folder, img + "_10.png")
        image_pair_list.append([left_image_path,right_image_path,disp_occ])
    return image_pair_list