import torch
from torchvision.transforms import Compose,Normalize,RandomCrop,RandomResizedCrop,Resize,RandomHorizontalFlip, ToTensor
from torchvision import transforms


def get_transforms():
    normalize = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transform = Compose([normalize])
    return transform

