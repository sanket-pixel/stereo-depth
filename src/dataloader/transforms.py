import torch
from torchvision.transforms import Compose,Normalize,RandomCrop,RandomResizedCrop, Resize, RandomHorizontalFlip, ToTensor
from torchvision import transforms
import random

def get_transforms():

    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize
    # randomly select from color jittering and blue
    color_changes = transforms.Compose([random.choice([transforms.ColorJitter(brightness=.5, hue=.3),
                                            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))]), transforms.ConvertImageDtype(torch.float32)])

    return color_changes, normalize

