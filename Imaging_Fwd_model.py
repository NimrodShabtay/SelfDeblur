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
        self.register_buffer('int_psi_values',
                             torch.Tensor(
                                 [psi_val for psi_val in self.psi_values if psi_val.int().float() == psi_val]))

    def psf_conv(self, sub_img, psf_kernels, conv_bias=None):
        """
        Notes:
             - unsqueeze(0) is done since F.conv2d expects [B, C, H, W] so adding Batch dim.
             - Casting the input to double to match the psf kernels weights type
        """
        pad_val = tuple([self.psf_kernel_size // 2 for _ in range(4)])
        pad_input = F.pad(sub_img, pad_val, mode='replicate').unsqueeze(0)
        blur_sub_img = F.conv2d(pad_input, psf_kernels.unsqueeze(1), bias=conv_bias, groups=self.num_psfs)
        return blur_sub_img.squeeze(0)

    def _extract_psf_kernels(self, depth):
        psi_ind = torch.isclose(depth.float(), self.psi_values,
                                atol=self.precomputed_params['psi_resolution'] / 2).nonzero(as_tuple=True)[0][0].item()
        return self.psf_kernels[:, :, :, psi_ind]

    def _quantize_psi_values(self, scaled_psi_map):
        quantized_scaled_psi_map = \
            torch.round(scaled_psi_map / self.precomputed_params['psi_resolution']) * self.precomputed_params[
                'psi_resolution']
        return quantized_scaled_psi_map

    def forward(self, x):
        rgb = x[:, :3, :, :]
        norm_psi = x[:, 3:, :, :]  # self._scale_value_to_psi_range(x[:, 3:, :, :])

        B, C, H, W = rgb.shape
        out = torch.zeros_like(rgb)
        for sample_idx in range(B):
            current_img_patch = rgb[sample_idx]  # [3, patch_h, patch_w]
            current_depth_patch = norm_psi[sample_idx]  # [15, patch_h, patch_w]
            blur_imgs = torch.zeros(len(self.int_psi_values), self.num_psfs, H, W, device=x.device)
            for psi_ind, psi in enumerate(self.int_psi_values):
                relevant_psf_kernels = self._extract_psf_kernels(psi)
                blur_sub_img = self.psf_conv(current_img_patch, relevant_psf_kernels)
                blur_imgs[psi_ind] = blur_sub_img

            out[sample_idx] = torch.einsum('ijkl, jmkl -> imkl', current_depth_patch.unsqueeze(0), blur_imgs)
        return out

    def forward_org(self, x):
        rgb = x[:, :3, :, :]
        norm_psi = x[:, 3:, :, :]
        quant_norm_psi = self._quantize_psi_values(norm_psi)

        B, _, h, w = rgb.shape
        out = torch.zeros(B, self.num_psfs, h, w, device=x.device)
        current_img_patch = rgb[0]  # [3, patch_h, patch_w]
        current_depth_patch = quant_norm_psi[0]  # [1, patch_h, patch_w]
        unique_depths = current_depth_patch.unique()
        blurred_img_pt = torch.zeros(self.num_psfs, h, w, device=x.device)
        overall_mask = torch.zeros(self.num_psfs, h, w, device=x.device)

        for unique_depth in unique_depths:
            relevant_psf_kernels = self._extract_psf_kernels(unique_depth)
            current_mask = (current_depth_patch == unique_depth).int()
            sub_img = current_img_patch * current_mask
            blur_sub_img = self.psf_conv(sub_img, relevant_psf_kernels)
            blurred_img_pt = blurred_img_pt + blur_sub_img
            # Construct normalization mask
            current_mask = self.psf_conv(current_mask.repeat(3, 1, 1).float(), relevant_psf_kernels)
            overall_mask = overall_mask + current_mask

        # Normalization over all depths in image patch
        overall_mask[overall_mask == 0] = 1
        blurred_img_pt = blurred_img_pt / overall_mask
        out[0] = torch.clip(blurred_img_pt, 0, 1)

        return out
