import torch
import numpy as np
import kornia.geometry.transform as dgm
import random


class Rotate:
    def __init__(self, n_trans, random_rotate=False):
        self.n_trans = n_trans
        self.random_rotate = random_rotate

    def apply(self, x):
        return rotate_dgm(x, self.n_trans, self.random_rotate)


def rotate_dgm(data, n_trans=5, random_rotate=False):
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 360)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))

    data = torch.cat([data if theta == 0 else dgm.rotate(data, torch.Tensor([theta]).type_as(data))
                      for theta in theta_list], dim=0)
    return data


class Shift:
    def __init__(self, n_trans, max_offset=0, direction=None):
        self.n_trans = n_trans
        self.max_offset = max_offset
        self.direction = direction

    def apply(self, x):
        return shift_random(x, self.n_trans, self.max_offset, self.direction)


def shift_random(x, n_trans=5, max_offset=0, direction=None):
    H, W = x.shape[-2], x.shape[-1]
    assert n_trans <= H - 1 and n_trans <= W - 1, 'n_shifts should less than {}'.format(H - 1)

    if max_offset == 0:
        shifts_row = random.sample(list(np.concatenate([-1 * np.arange(1, H), np.arange(1, H)])), n_trans)
        shifts_col = random.sample(list(np.concatenate([-1 * np.arange(1, W), np.arange(1, W)])), n_trans)
    else:
        assert max_offset <= min(H, W), 'max_offset must be less than min(H,W)'
        shifts_row = random.sample(list(np.concatenate([-1 * np.arange(1, max_offset), np.arange(1, max_offset)])),
                                   n_trans)
        shifts_col = random.sample(list(np.concatenate([-1 * np.arange(1, max_offset), np.arange(1, max_offset)])),
                                   n_trans)

    x = torch.cat([x if n_trans == 0 else torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x) for sx, sy in
                   zip(shifts_row, shifts_col)], dim=0)
    return x
