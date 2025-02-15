
from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from networks.skip import skip
from networks.fcn import fcn
import cv2
import torch
import torch.optim
from torch.autograd import Variable
import glob
from skimage.io import imread
from skimage.io import imsave
from PIL import Image
import warnings
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils.common_utils import *
from SSIM import SSIM

import wandb
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision.io import read_image
from utils.psf_utils import extract_psf_from_matlab


parser = argparse.ArgumentParser()
parser.add_argument('--num_iter', type=int, default=5000, help='number of epochs of training')
parser.add_argument('--img_size', type=int, default=[256, 256], help='size of each image dimension')
parser.add_argument('--kernel_size', type=int, default=[31, 31], help='size of blur kernel [height, width]')
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

input_source = glob.glob(os.path.join(opt.data_path, 'output_downscaled/rgb', '*.png'))
input_source.sort()

sharp_source = glob.glob(os.path.join(opt.data_path, 'Images', '*.png'))
sharp_source.sort()

save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)

files_source = input_source  # sorted(glob.glob('./datasets/real/*.jpg'))  #
# start #image
idx = 0
for f in files_source[idx:idx+1]:
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

    img, y, cb, cr = readimg(path_to_image)
    y = np.float32(y / 255.0)
    y = np.expand_dims(y, 0)
    img_size = y.shape

    sharp_img, _, _, _ = readimg(sharp_source[idx])
    sharp_img_np = np.float32(sharp_img / 255.0)
    # sharp_img_np = np.expand_dims(sharp_img, 0)

    # psf_mat_file_path = '/mnt5/nimrod/SelfDeblur/datasets/synt/city_cls/data.mat'
    # out_k_m_ref = extract_psf_from_matlab(psf_mat_file_path, dtype=dtype)
    print(imgname)
    # ######################################################################

    padw, padh = opt.kernel_size[0]-1, opt.kernel_size[1]-1
    opt.img_size[0], opt.img_size[1] = img_size[1]+padw, img_size[2]+padh
    #y = y[:, padh//2:img_size[1]-padh//2, padw//2:img_size[2]-padw//2]
    y = np_to_torch(y).type(dtype)

    input_depth = 8

    net_input = get_noise(input_depth, INPUT, (opt.img_size[0], opt.img_size[1])).type(dtype)

    net = skip(input_depth, 1,
               num_channels_down=[128, 128, 128, 128, 128],
               num_channels_up=[128, 128, 128, 128, 128],
               num_channels_skip=[16, 16, 16, 16, 16],
               upsample_mode='bilinear',
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

    net = net.type(dtype)

    n_k = 200
    net_input_kernel = get_noise(n_k, INPUT, (1, 1)).type(dtype)
    net_input_kernel.squeeze_()

    net_kernel = fcn(n_k, opt.kernel_size[0] * opt.kernel_size[1])
    net_kernel = net_kernel.type(dtype)

    # Losses
    mse = torch.nn.MSELoss().type(dtype)
    ssim = SSIM().type(dtype)

    # optimizer
    optimizer = torch.optim.Adam([{'params': net.parameters()}, {'params': net_kernel.parameters(), 'lr': 1e-4}], lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[1600, 1900, 2200], gamma=0.5)  # learning rates

    # initilization inputs
    net_input_saved = net_input.detach().clone()
    net_input_kernel_saved = net_input_kernel.detach().clone()

    log_config = {
        'input depth': input_depth,
        'losses': ', '.join(map(str, [type(mse).__name__, type(ssim).__name__])),
        'initial LR': LR,
        'reg_noise_std': reg_noise_std,
    }
    run = wandb.init(project="Dip-Defocus",
                     entity="impliciteam",
                     tags=['deblurring', 'unknown kernel', 'reference'],
                     name='Deblurring - Ycbcr',
                     job_type='train',
                     mode='offline',
                     save_code=True,
                     config=log_config,
                     notes='Psi = 4 for all map'
                     )

    wandb.run.log_code(".")

    ### start SelfDeblur
    for step in tqdm(range(num_iter)):

        # input regularization
        net_input = net_input_saved + reg_noise_std * torch.zeros(net_input_saved.shape).type_as(
            net_input_saved.data).normal_()
        net_input_kernel = net_input_kernel_saved + reg_noise_std * \
                           torch.zeros(net_input_kernel_saved.shape).type_as(net_input_kernel_saved.data).normal_()

        # change the learning rate
        scheduler.step(step)
        optimizer.zero_grad()

        # get the network output
        out_x = net(net_input)
        out_k = net_kernel(net_input_kernel)

        out_k_m = out_k.view(-1, 1, opt.kernel_size[0], opt.kernel_size[1])

        # print(out_k_m)
        out_y = nn.functional.conv2d(out_x, out_k_m, padding=0, bias=None)

        y_size = out_y.shape
        cropw = y_size[2]-img_size[1]
        croph = y_size[3]-img_size[2]
        out_y = out_y[:,:,cropw//2:cropw//2+img_size[1],croph//2:croph//2+img_size[2]]

        if step < 500:
            total_loss = mse(out_y, y)
        else:
            total_loss = 1 - ssim(out_y, y)

        wandb.log({'Loss': total_loss.item()})
        total_loss.backward()
        optimizer.step()

        if (step + 1) % opt.save_frequency == 0:
            # print('Iteration %05d' %(step+1))
            H, W, _ = img.shape
            out_x_np = out_x[0].permute(1, 2, 0).detach().cpu().numpy()
            out_x_np = out_x_np[padh // 2: -(padh // 2), padw // 2: -(padw // 2), 0]
            out_x_np = np.uint8(255 * out_x_np)
            out_x_np = cv2.merge([out_x_np, cr, cb])
            out_x_np = cv2.cvtColor(out_x_np, cv2.COLOR_YCrCb2BGR)
            out_x_np = out_x_np / 255.0
            img_np = img

            out_y_np = out_y[0].permute(1, 2, 0).detach().cpu().numpy()
            out_y_np = np.uint8(255 * out_y_np)
            out_y_np = cv2.merge([out_y_np[:, :, 0], cr, cb])
            out_y_np = cv2.cvtColor(out_y_np, cv2.COLOR_YCrCb2BGR)

            blur_psnr = psnr(out_y_np, img_np)
            sharp_psnr = psnr(out_x_np, sharp_img_np)
            # sharp_psnr = -1.

            sharp_img_to_log = np.zeros((H, 2 * W, 3), dtype=np.float)
            blur_img_to_log = np.zeros((H, 2 * W, 3), dtype=np.float)

            sharp_img_to_log[:, :W, :] = sharp_img_np
            # sharp_img_to_log[:, :W, :] = np.zeros_like(out_x_np)
            sharp_img_to_log[:, W:, :] = out_x_np

            blur_img_to_log[:, :W, :] = img_np
            blur_img_to_log[:, W:, :] = out_y_np

            wandb.log(
                {'Sharp Img':
                     wandb.Image(sharp_img_to_log, caption='PSNR: {}'.format(sharp_psnr)),
                 'Blur Img':
                     wandb.Image(blur_img_to_log, caption='PSNR: {}'.format(blur_psnr))}, commit=False)
            wandb.log({'blur psnr': blur_psnr, 'sharp psnr': sharp_psnr}, commit=True)


            # save_path = os.path.join(opt.save_path, '%s_x.png' % imgname)
            # out_x_np = torch_to_np(out_x)
            # out_x_np = out_x_np.squeeze()
            # cropw, croph = padw, padh
            # out_x_np = out_x_np[cropw//2:cropw//2+img_size[1], croph//2:croph//2+img_size[2]]
            # out_x_np = np.uint8(255 * out_x_np)
            # out_x_np = cv2.merge([out_x_np, cr, cb])
            # out_x_np = cv2.cvtColor(out_x_np, cv2.COLOR_YCrCb2BGR)
            # cv2.imwrite(save_path, out_x_np)
            #
            # save_path = os.path.join(opt.save_path, '%s_k.png' % imgname)
            # out_k_np = torch_to_np(out_k_m)
            # out_k_np = out_k_np.squeeze()
            # out_k_np /= np.max(out_k_np)
            # imsave(save_path, out_k_np)
            #
            # torch.save(net, os.path.join(opt.save_path, "%s_xnet.pth" % imgname))
            # torch.save(net_kernel, os.path.join(opt.save_path, "%s_knet.pth" % imgname))
