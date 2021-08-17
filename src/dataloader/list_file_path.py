import os

def get_image_pair_names(data_path):
    image_pair_list = []
    for sub_folder in os.listdir(data_path):
        data_path_1 = os.path.join(data_path, sub_folder, "disparity")
        for item in os.listdir(data_path_1):
            disparity_path = os.path.join(data_path_1, item)
            item_name = item[:-4]
            left_image_path = os.path.join(data_path,sub_folder, "RGB_cleanpass", "left", item_name + '.png')
            right_image_path = os.path.join(data_path, sub_folder, "RGB_cleanpass", "right", item_name + '.png')
            image_pair_list.append([left_image_path, right_image_path, disparity_path])
    return image_pair_list


