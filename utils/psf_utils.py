import torch
from scipy.io import loadmat
import numpy as np


def extract_psf_from_matlab(path, dtype, num_psfs=3, psf_kernel_size=71):
    mat_file = loadmat(path)['data'][0][0]
    precomputed_params = {
        'radius': np.asscalar(mat_file[0]),
        'ref_wl': np.asscalar(mat_file[1]),
        'focus_point': np.asscalar(mat_file[2]),
        'Psi': np.asarray(mat_file[3])[0],
        'min_dist': np.asscalar(mat_file[4]),
        'psf_kernels': mat_file[5].reshape(
            num_psfs, psf_kernel_size, psf_kernel_size, -1),
        'map_interval': np.asscalar(mat_file[6]),
        'num_psi_classes': np.asscalar(mat_file[7]),
        'downsample_factor': np.asscalar(mat_file[8]),
        'dn_fs': np.asscalar(mat_file[9]),
    }
    return torch.from_numpy(precomputed_params['psf_kernels'][:, :, :, 0]).type(dtype)  # Extracting only the PSF filters corresponds to Psi=-4


TAG_FLOAT = 202021.25


# Depth IO taken from https://github.com/YotYot/CalibrationNet/blob/master/models/sintel_io.py
def depth_read(filename):
    """ Read depth data from file, return as numpy array. """
    f = open(filename, 'rb')
    check = np.fromfile(f, dtype=np.float32, count=1)[0]
    assert check == TAG_FLOAT, \
        'depth_read:: Wrong tag in flow file (should be: {0}, is: {1}). Big-endian machine? '.format(TAG_FLOAT, check)
    width = np.fromfile(f, dtype=np.int32, count=1)[0]
    height = np.fromfile(f, dtype=np.int32, count=1)[0]
    size = width * height
    assert width > 0 and height > 0 and 1 < size < 100000000, \
        ' depth_read:: Wrong input size (width = {0}, height = {1}).'.format(width, height)
    depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def scale_psi(norm_psi_map, psi_min=-4., psi_max=10.):
    psi_range = (psi_max - psi_min)
    psi_range_center_sym = psi_min + (psi_range / 2)
    scaled_psi_map = (norm_psi_map * psi_range / 2) + psi_range_center_sym
    return scaled_psi_map


def convert_weights_to_psi(depth_as_weights, psi_min, psi_max):
    int_psi_vals = np.array([ps_v for ps_v in range(psi_min, psi_max + 1, 1)])
    psi_from_weights = depth_as_weights @ int_psi_vals
    return psi_from_weights
