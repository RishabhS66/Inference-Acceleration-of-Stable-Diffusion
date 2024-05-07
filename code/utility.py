import torch
from torch import nn
from torch.nn import functional as F

def percentage_pruned(model):
    total_zeros = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_zeros += torch.sum(param == 0).item()
        total_params += param.numel()

    return total_params - total_zeros, (total_zeros / total_params) * 100