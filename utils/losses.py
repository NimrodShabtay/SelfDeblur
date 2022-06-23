import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class StdLoss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(StdLoss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)

    def forward(self, x):
        return self.mse(F.conv2d(x, self.image), F.conv2d(x, self.blur))


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
