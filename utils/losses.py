import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import Sobel


# Taken from: https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        pixel_dif1 = x[..., 1:, :] - x[..., :-1, :]
        pixel_dif2 = x[..., :, 1:] - x[..., :, :-1]

        res1 = pixel_dif1.abs().mean()
        res2 = pixel_dif2.abs().mean()

        return res1 + res2


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.op = Sobel()
        self.criterion = nn.L1Loss()

    def forward(self, map, img):
        img_grad = self.op(img)
        map_grad = self.op(map.repeat(1, 3, 1, 1))
        return self.criterion(img_grad, map_grad)


class ImageStatsLoss(nn.Module):
    def __init__(self):
        super(ImageStatsLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, sharp_img, ref_blur_img):
        ref_per_channel_mean = torch.mean(ref_blur_img, dim=[-2, -1])
        sharp_img_per_channel_mean = torch.mean(sharp_img, dim=[-2, -1])
        return self.criterion(ref_per_channel_mean, sharp_img_per_channel_mean)
