
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

import utils.psf_utils
from networks.skip import skip
import glob
import matplotlib.pyplot as plt

import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM

from torchvision.utils import save_image
from torchvision.io import read_image
from skimage.metrics import peak_signal_noise_ratio as psnr
import wandb

from Imaging_Fwd_model import FwdModel
from utils.psf_utils import depth_read
from utils.losses import TVLoss, GradLoss
from kornia.losses import InverseDepthSmoothnessLoss


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=3000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 512], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[71, 71], help='size of blur kernel [height, width]')
parser.add_argument('--data_path', type=str, default="datasets/real", help='path to blurry image')
parser.add_argument('--save_path', type=str, default="results/synt/", help='path to deblurring results')
parser.add_argument('--save_frequency', type=int, default=100, help='lfrequency to save results')
parser.add_argument('--gpu', default='0')

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
# print(opt)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

warnings.filterwarnings("ignore")

input_source = glob.glob(os.path.join(opt.data_path, 'output_downscaled/rgb',  '*.png'))
input_source.sort()

sharp_source = glob.glob(os.path.join(opt.data_path, 'Images', '*.png'))
sharp_source.sort()

psi_maps_source = glob.glob(os.path.join(opt.data_path, 'output_downscaled/GT', '*.dpt'))
psi_maps_source.sort()

save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

index = 86
# start #image
for f in input_source[index:index+1]:
    INPUT = 'noise'
    pad = 'reflection'
    LR = 0.01
    num_iter = opt.num_iter
    reg_noise_std = 0.001

    path_to_image = f
    imgname = os.path.basename(f)
    imgname = os.path.splitext(imgname)[0]

    if imgname.find('fish') != -1:
        opt.kernel_size = [41, 41]
    if imgname.find('flower') != -1:
        opt.kernel_size = [25, 25]
    if imgname.find('house') != -1:
        opt.kernel_size = [51, 51]
    if imgname.find('maskImg') != -1:
        opt.kernel_size = [71, 71]

    PSI_CLASSES = 15
    DEPTH_MIN = -4.0
    DEPTH_MAX = 10.0

    img = read_image(path_to_image).type(dtype)
    img /= 255.0
    img_size = img.shape

    # sharp_img = read_image(sharp_source[index]).float()
    sharp_img = torchvision.transforms.ToTensor()(Image.open(sharp_source[index]).convert('RGB'))
    # sharp_img /= 255.0
    sharp_img_np = sharp_img.permute(1, 2, 0).numpy()

    psi_map_ref = depth_read(psi_maps_source[index])
    psi_map_ref = np.expand_dims(psi_map_ref, -1)

    psi_map_ref = (psi_map_ref - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

    # psi_gt = torch.from_numpy(psi_map_ref).permute(2, 0, 1).unsqueeze(0).type(dtype)
    print(imgname)
    # ######################################################################

    padw, padh = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padh, img_size[2]+padw

    input_depth = 32  # Consider split to depth and img with different depths
    img.unsqueeze_(0)

    net_img_input = get_noise(input_depth, INPUT, (img_size[1], img_size[2])).type(dtype)
    net_psi_input = get_noise(input_depth, INPUT, (img_size[1], img_size[2])).type(dtype)

    net_img = skip(input_depth, 3,
                   num_channels_down=[128, 128, 128, 128, 128],
                   num_channels_up=[128, 128, 128, 128, 128],
                   num_channels_skip=[16, 16, 16, 16, 16],
                   upsample_mode='bilinear',
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net_img = net_img.type(dtype)

    net_psi = skip(input_depth, 1,
                   num_channels_down=[128, 128, 128, 128, 128],
                   num_channels_up=[128, 128, 128, 128, 128],
                   num_channels_skip=[16, 16, 16, 16, 16],
                   upsample_mode='bilinear',
                   need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net_psi = net_psi.type(dtype)

    fwd_model = FwdModel()
    fwd_model.load_pre_compute_data('/mnt5/nimrod/depth_from_defocus/data/mask_synt_data/headbutt/output/data.mat')
    fwd_model.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)
    tv_loss = TVLoss().type(dtype)
    grad_loss = GradLoss().type(dtype)
    # depth_smoothness_loss = InverseDepthSmoothnessLoss().type(dtype)
    tv_weight = 0.2
    grad_weight = 1

    # optimizer
    optimizer = torch.optim.Adam([{'params': net_img.parameters()}, {'params': net_psi.parameters()}], lr=LR)
    # optimizer = torch.optim.Adam([{'params': net_img.parameters()}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[1600, 1900, 2200], gamma=0.5)  # learning rates

    # initilization inputs
    net_img_input_saved = net_img_input.detach().clone()
    net_psi_input_saved = net_psi_input.detach().clone()

    log_config = {
        'input depth': input_depth,
        'losses': ', '.join(map(str, [type(mse).__name__, type(ssim).__name__])),
        'initial LR': LR,
        'reg_noise_std': reg_noise_std,
        'Psi Classes': PSI_CLASSES
    }
    run = wandb.init(project="Dip-Defocus",
                     entity="impliciteam",
                     tags=['defocus'],
                     name='Defocus - - Grad + TV regs',
                     job_type='train',
                     mode='online',
                     save_code=True,
                     config=log_config,
                     notes='Defocus - Grad + TV regs on depth'
                     )

    wandb.run.log_code(".")

    ### start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_img_input = net_img_input_saved + reg_noise_std * torch.zeros(net_img_input_saved.shape).type_as(
            net_img_input_saved.data).normal_()

        net_psi_input = net_psi_input_saved + reg_noise_std * torch.zeros(net_psi_input_saved.shape).type_as(
            net_psi_input_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net_img(net_img_input)
        out_psi = net_psi(net_psi_input)
        scaled_psi = utils.psf_utils.scale_psi(out_psi)
        fwd_model_inputs = torch.cat([out_x, scaled_psi], dim=1)
        out_rgb = fwd_model(fwd_model_inputs)

        rgb_size = out_rgb.shape
        cropw = rgb_size[2] - img_size[1]
        croph = rgb_size[3] - img_size[2]
        out_rgb = out_rgb[:, :, cropw // 2:cropw // 2 + img_size[1], croph // 2:croph // 2 + img_size[2]]

        if step < 500:
            total_loss = mse(out_rgb, img)
        else:
            total_loss = 1 - ssim(out_rgb, img)

        if step > 300:
            reg_loss = tv_weight * tv_loss(out_psi) + grad_weight * grad_loss(out_psi, out_x)
            total_loss += reg_loss

        wandb.log({'Loss': total_loss.item()})
        total_loss.backward()
        optimizer.step()

        if step % opt.save_frequency == 0:
            _, C, H, W = out_rgb.shape
            out_x_np = out_x[0].permute(1, 2, 0).detach().cpu().numpy()
            out_rgb_np = out_rgb[0].permute(1, 2, 0).detach().cpu().numpy()
            img_np = img[0].permute(1, 2, 0).cpu().numpy()
            psi_np = out_psi[0].permute(1, 2, 0).cpu().detach().numpy()

            blur_psnr = psnr(out_rgb_np, img_np)
            sharp_psnr = psnr(out_x_np, sharp_img_np)
            depth_psnr = psnr(psi_np, psi_map_ref)

            sharp_img_to_log = np.zeros((H, 2 * W, 3), dtype=np.float)
            blur_img_to_log = np.zeros((H, 2 * W, 3), dtype=np.float)
            psi_map_to_log = np.zeros((H, 2 * W, 1), dtype=np.float)

            sharp_img_to_log[:, :W, :] = sharp_img_np
            sharp_img_to_log[:, W:, :] = out_x_np

            blur_img_to_log[:, :W, :] = img_np
            blur_img_to_log[:, W:, :] = out_rgb_np

            psi_map_to_log[:, :W, :] = psi_map_ref
            psi_map_to_log[:, W:, :] = psi_np

            fig = plt.figure(figsize=(12, 6))
            im = plt.imshow(psi_np)
            plt.colorbar(im, fraction=0.046, pad=0.04)

            wandb.log(
                {'Sharp Img':
                     wandb.Image(sharp_img_to_log, caption='PSNR: {}'.format(sharp_psnr)),
                 'Blur Img':
                     wandb.Image(blur_img_to_log, caption='PSNR: {}'.format(blur_psnr)),
                 'Psi Map':
                     [wandb.Image(psi_map_to_log, caption='PSNR: {}'.format(depth_psnr)),
                      wandb.Image(fig, caption='Depth for Visual purposes')],
                 }, commit=False)

            wandb.log({'blur psnr': blur_psnr, 'sharp psnr': sharp_psnr, 'depth psnr': depth_psnr}, commit=False)
            # save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
            # out_x = out_x.squeeze()
            # cropw, croph = padw, padh
            # out_x = out_x[:, cropw // 2:cropw // 2 + img_size[1], croph // 2:croph // 2 + img_size[2]]
            # save_image(out_x, save_path)
            #
            # torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))

