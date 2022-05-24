""" File for training the Blend GAN (autoencoder)"""
import os
from datetime import datetime
from os.path import join
import pathlib
from tqdm import tqdm
import argparse

import repackage
repackage.up()

import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
from torchvision.transforms import Pad
from torchvision.utils import make_grid
import repackage
repackage.up()

from imagenet.config_gpgan import get_cfg_gp_gan_defaults
from imagenet.models import BlendGAN, CGN
from imagenet.models.gp_gan import Encoder
from shared.losses import *
from utils import Optimizers


def save_sample_sheet(blend_gan, u_fixed, sample_path, ep_str):
    blend_gan.eval()
    dev = u_fixed.to(cgn.get_device())
    ys = [15, 251, 330, 382, 385, 483, 559, 751, 938, 947, 999]

    to_save = []
    with torch.no_grad():
        for y in ys:
            # generate
            y_vec = blend_gan.get_class_vec(y, sz=1)
            inp = (u_fixed.to(dev), y_vec.to(dev), cgn.truncation)
            x_gt, mask, premask, foreground, background, bg_mask = cgn(inp)
            x_gen = mask * foreground + (1 - mask) * background

            # build class grid
            to_plot = [premask, foreground, background, x_gen, x_gt]
            grid = make_grid(torch.cat(to_plot).detach().cpu(),
                             nrow=len(to_plot), padding=2, normalize=True)

            # add unnormalized mask
            mask = Pad(2)(mask[0].repeat(3, 1, 1)).detach().cpu()
            grid = torch.cat([mask, grid], 2)

            # save to disk
            to_save.append(grid)
            del to_plot, mask, premask, foreground, background, x_gen, x_gt

    # save the image
    path = join(sample_path, f'cls_sheet_' + ep_str + '.png')
    torchvision.utils.save_image(torch.cat(to_save, 1), path)
    blend_gan.train()

def save_sample_single(blend_gan, u_fixed, sample_path, ep_str):
    blend_gan.eval()
    dev = u_fixed.to(blend_gan.get_device())

    ys = [15, 251, 330, 382, 385, 483, 559, 751, 938, 947, 999]
    with torch.no_grad():
        for y in ys:
            # generate
            y_vec = blend_gan.get_class_vec(y, sz=1)
            inp = (u_fixed.to(dev), y_vec.to(dev), blend_gan.truncation)
            _, mask, premask, foreground, background, _ = blend_gan(inp)
            x_gen = mask * foreground + (1 - mask) * background

            # save_images
            path = join(sample_path, f'{y}_1_premask_' + ep_str + '.png')
            torchvision.utils.save_image(premask, path, normalize=True)
            path = join(sample_path, f'{y}_2_mask_' + ep_str + '.png')
            torchvision.utils.save_image(mask, path, normalize=True)
            path = join(sample_path, f'{y}_3_texture_' + ep_str + '.png')
            torchvision.utils.save_image(foreground, path, normalize=True)
            path = join(sample_path, f'{y}_4_bgs_' + ep_str + '.png')
            torchvision.utils.save_image(background, path, normalize=True)
            path = join(sample_path, f'{y}_5_gen_ims_' + ep_str + '.png')
            torchvision.utils.save_image(x_gen, path, normalize=True)
    blend_gan.train()

def fit(cfg, blend_gan, discriminator, cgn, opts, losses):
    """ Training the blend_gan
    Args:
        - cfg: configurations
        - blend_gan: autoencoder model
        - discriminator: discriminator to get the adverserial loss
        - cgn: the CGN module used for Imagenet in the orignial paper, to be used as input and ground truth for blend_gan
        - opts: optimisers
        - losses: reconstruction (MSE) & adverserial
    """
    # total number of episodes, accounted for batch accumulation
    episodes = cfg.TRAIN.EPISODES
    episodes *= cfg.TRAIN.BATCH_ACC

    # directories for experiments and storing results
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # if cfg.WEIGHTS_PATH:
    #     weights_path = str(pathlib.Path(cfg.WEIGHTS_PATH).parent)
    #     start_ep = int(pathlib.Path(cfg.WEIGHTS_PATH).stem[3:])
    #     sample_path = weights_path.replace('weights', 'samples')
    #     ep_range = (start_ep, start_ep + episodes)
    # else:
    #     model_path = join('imagenet', 'experiments',
    #                       f'cgn_{time_str}_{cfg.MODEL_NAME}')
    #     weights_path = join(model_path, 'weights')
    #     sample_path = join(model_path, 'samples')
    #     pathlib.Path(weights_path).mkdir(parents=True, exist_ok=True)
    #     pathlib.Path(sample_path).mkdir(parents=True, exist_ok=True)
    #     ep_range = (0, episodes)
    ep_range = (0, episodes)

    # Training loop
    blend_gan.train()
    L_l2, L_adv = losses

    pbar = tqdm(range(*ep_range))
    print(f"training for  {ep_range[1]} episodes...") # ToDo: remove
    for i, ep in enumerate(pbar):

        """ Training the blend_gan """
        # generate x (copy_paste_compose)
        # generate x_gt (from cGAN)

        # get the low resolution, well-blended, semantic & colour accurate output x_l


        # pass inputs into discriminator ToDo: which model? 


        # get losses

        """ Training the discriminator """

        break 
        # ToDo:



def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init model
    blend_gan = BlendGAN()
    # ToDo: init discriminator
    discriminator = Encoder()

    # init cgn
    cgn = CGN(
        batch_sz=cfg.TRAIN.BATCH_SZ,
        truncation=cfg.MODEL.TRUNCATION,
        pretrained=True,
    )

    if cfg.CGN_WEIGHTS_PATH:
        print(f"Loading CGN weights from {cfg.CGN_WEIGHTS_PATH}")
        weights = torch.load(cfg.CGN_WEIGHTS_PATH, map_location=torch.device(device))
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        cgn.load_state_dict(weights)
    
    # optimizers
    opts = Optimizers()
    opts.set('blend_gan', blend_gan, lr=cfg.LR.BGAN)
    # ToDo:  opts.set('discriminator', discriminator, ...)
    # push to device and train
    
    blend_gan = blend_gan.to(device)
    #discriminator = discriminator.to(device)

    #losses
    L_l2 = ReconstructionLoss(mode='l2', loss_weight=cfg.LAMBDA.L2)
    L_adv = torch.nn.MSELoss()  # ToDo: add correct loss
    losses = (L_l2, L_adv)
    # ToDo: L_adv

    fit(cfg, blend_gan, discriminator, cgn, opts, losses)  # train models


  # ToDo: check if correct
def merge_args_and_cfg(args, cfg):
    cfg.MODEL_NAME = args.model_name
    cfg.CGN_WEIGHTS_PATH = args.weights_path

    # cfg.LOG.SAMPLED_FIXED_NOISE = args.sampled_fixed_noise
    # cfg.LOG.SAVE_SINGLES = args.save_singles
    # cfg.LOG.SAVE_ITER = args.save_iter
    # cfg.LOG.LOSSES = args.log_losses

    cfg.TRAIN.EPOCHS = args.epochs
    cfg.TRAIN.BATCH_SZ = args.batch_sz
    cfg.TRAIN.BATCH_ACC = args.batch_acc

    # cfg.MODEL.TRUNCATION = args.truncation
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='tmp',
                        help='Weights and samples will be saved under experiments/model_name')
    parser.add_argument('--weights_path', default='imagenet/weights/cgn.pth',
                        help='provide path to continue training')
    parser.add_argument('--epochs', type=int, default=300,
                        help="We don't do dataloading, hence, one episode = one gradient update.")
    parser.add_argument('--batch_sz', type=int, default=1,
                        help='Batch size, use in conjunciton with batch_acc')
    parser.add_argument('--batch_acc', type=int, default=4000,
                        help='pseudo_batch_size = batch_acc*batch size')

    # ToDo: add more
    args = parser.parse_args()

    cfg = get_cfg_gp_gan_defaults()
    cfg = merge_args_and_cfg(args, cfg)

    print(cfg)
    main(cfg)