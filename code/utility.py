import torch
import numpy as np
import os
from PIL import Image
from torchvision.transforms import functional as tvf

def percentage_pruned(model):
    total_zeros = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_zeros += torch.sum(param == 0).item()
        total_params += param.numel()

    return total_params - total_zeros, (total_zeros / total_params) * 100


def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return tvf.center_crop(image, (256, 256))

def get_real_images():
    dataset_path = "./data/sample-imagenet-images"
    image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])

    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    real_images = torch.cat([preprocess_image(image) for image in real_images])
    return real_images