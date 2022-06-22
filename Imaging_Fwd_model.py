import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import tqdm


class FwdModel(nn.Module):
    def __init__(self, in_ftrs=4, out_ftrs=3, psf_kernel_size=71, num_psfs=3, psi_range=(-4., 10.)):
        super(FwdModel, self).__init__()
        self.out_rgb = out_ftrs
        self.psf_kernel_size = psf_kernel_size
        self.num_psfs = num_psfs
        self.precomputed_params = {}
        self.op = nn.Linear(in_ftrs, out_ftrs)  # Placeholder
        self.psi_range = psi_range

    def load_pre_compute_data(self, precompute_data_path):
        from scipy.io import loadmat
        mat_file = loadmat(precompute_data_path)['data'][0][0]
        precomputed_params = {
            'radius': np.asscalar(mat_file[0]),
            'ref_wl': np.asscalar(mat_file[1]),
            'focus_point': np.asscalar(mat_file[2]),
            'Psi': np.asarray(mat_file[3])[0],
            'min_dist': np.asscalar(mat_file[4]),
            'psf_kernels': mat_file[5].reshape(
                self.num_psfs, self.psf_kernel_size, self.psf_kernel_size, -1),
            'map_interval': np.asscalar(mat_file[6]),
            'num_psi_classes': np.asscalar(mat_file[7]),
            'downsample_factor': np.asscalar(mat_file[8]),
            'dn_fs': np.asscalar(mat_file[9]),
        }
        precomputed_params.update({'psi_resolution':
                                       float(precomputed_params['Psi'][-1] - precomputed_params['Psi'][-2])})
        self.precomputed_params = precomputed_params

        self.register_buffer('psf_kernels', torch.from_numpy(precomputed_params['psf_kernels']))
        self.register_buffer('psi_values', torch.from_numpy(precomputed_params['Psi']).float())

    def psf_conv(self, sub_img, psf_kernels, conv_bias=None):
        """
        Notes:
             - unsqueeze(0) is done since F.conv2d expects [B, C, H, W] so adding Batch dim.
             - Casting the input to double to match the psf kernels weights type
        """
        pad_input = sub_img.unsqueeze(0)
        blur_sub_img = F.conv2d(pad_input, psf_kernels.unsqueeze(1), bias=conv_bias, groups=self.num_psfs)
        return blur_sub_img.squeeze(0)

    def _extract_psf_kernels(self, depth):
        psi_ind = torch.isclose(depth.float(), self.psi_values,
                                atol=self.precomputed_params['psi_resolution'] / 2).nonzero(as_tuple=True)[0][0].item()
        return self.psf_kernels[:, :, :, psi_ind]

    def forward(self, x, psi_map):
        rgb_patches = x
        depth_map_patches = psi_map

        num_samples, _, patch_h, patch_w = rgb_patches.shape
        spatial_dim_w = patch_w - (self.psf_kernel_size - 1)
        spatial_dim_h = patch_h - (self.psf_kernel_size - 1)
        out = torch.zeros(num_samples, self.num_psfs, spatial_dim_h, spatial_dim_w, device=x.device)
        for sample in range(num_samples):
            current_img_patch = rgb_patches[sample]  # [3, patch_h, patch_w]
            current_depth_patch = depth_map_patches[sample].clone()  # [3, patch_h, patch_w]
            current_depth_patch += self.psi_range[0]  # Shift values from classes [0, 14] to Psi [-4, 10]
            unique_depths = current_depth_patch.unique()
            blurred_img_pt = torch.zeros(self.num_psfs, spatial_dim_h, spatial_dim_w, device=x.device)
            overall_mask = torch.zeros(self.num_psfs, spatial_dim_h, spatial_dim_w, device=x.device)

            for unique_depth in unique_depths:
                relevant_psf_kernels = self._extract_psf_kernels(unique_depth)
                current_mask = (current_depth_patch == unique_depth).int()
                current_mask = F.pad(current_mask, [self.psf_kernel_size // 2 for _ in range(4)],
                                     mode='constant', value=1)
                sub_img = current_img_patch * current_mask
                blur_sub_img = self.psf_conv(sub_img, relevant_psf_kernels)
                blurred_img_pt += blur_sub_img
                # Construct normalization mask
                current_mask = self.psf_conv(current_mask.repeat(3, 1, 1).float(), relevant_psf_kernels)
                overall_mask += current_mask

            # Normalization over all depths in image patch
            overall_mask[overall_mask == 0] = 1
            blurred_img_pt /= overall_mask
            blurred_img_pt = torch.clip(blurred_img_pt, 0, 1)
            out[sample] = blurred_img_pt

        return out
