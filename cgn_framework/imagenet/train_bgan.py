""" File for training the Blend GAN (autoencoder)"""
import os
from datetime import datetime
from os.path import join
import pathlib
from tqdm import tqdm
import argparse
import csv

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


def save_sample_sheet(blend_gan, cgn, sample_path, ep_str):
    
    blend_gan.eval()

    to_save = []
    with torch.no_grad():
        #for y in ys:
            # generate
            # y_vec = blend_gan.get_class_vec(y, sz=1)
            # inp = (u_fixed.to(dev), y_vec.to(dev), cgn.truncation)
        x_gt, mask, premask, foreground, background, bg_mask = cgn()
        x_gen = mask * foreground + (1 - mask) * background
        x_l = blend_gan(x_gen)
        # build class grid
        to_plot = [x_gen, x_l, x_gt]
        grid = make_grid(torch.cat(to_plot).detach().cpu(),
                             nrow=len(to_plot), padding=2, normalize=True)

        # save to disk
        to_save.append(grid)
        del to_plot, mask, premask, foreground, background, x_gen, x_gt

    #save the image
    path = join(sample_path, f'cls_sheet_' + ep_str + '.png')
    torchvision.utils.save_image(torch.cat(to_save, 1), path)
    blend_gan.train()

def save_sample_single(blend_gan, cgn, sample_path, ep_str):
    
    blend_gan.eval()
    # ys = [15, 251, 330, 382, 385, 483, 559, 751, 938, 947, 999]
    with torch.no_grad():
        #for y in ys:
        # generate
        _, mask, _, foreground, background, _ = cgn()
        x = mask * foreground + (1 - mask) * background
        x_l = blend_gan(x)

        # save_images # ToDo: consider adding image classes
        path = join(sample_path, f'1_composite_img_x_' + ep_str + '.png')
        torchvision.utils.save_image(x, path, normalize=True)
        path = join(sample_path, f'_2_refined_img_xl_' + ep_str + '.png')
        torchvision.utils.save_image(x_l, path, normalize=True)
    blend_gan.train()


def fit(cfg, blend_gan, discriminator, cgn, opts, losses, device=None):
    """ Training the blend_gan
    Args:
        - cfg: configurations
        - blend_gan: autoencoder model
        - discriminator: discriminator to get the adverserial loss
        - cgn: the CGN module used for Imagenet in the orignial paper, to be used as input and ground truth for blend_gan
        - opts: optimisers
        - losses: reconstruction (MSE) & adverserial
    """

    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # total number of episodes, accounted for batch accumulation
    episodes = cfg.TRAIN.EPOCHS
    episodes *= cfg.TRAIN.BATCH_ACC

    # directories for experiments and storing results
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    model_path = join('imagenet', 'experiments',
                          f'bgn_{time_str}_{cfg.MODEL_NAME}')
    weights_path = join(model_path, 'weights')
    sample_path = join(model_path, 'samples')
    loss_path = join(model_path, 'losses')

    pathlib.Path(weights_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(sample_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(loss_path).mkdir(parents=True, exist_ok=True)
    ep_range = (0, episodes)

    # Training loop
    blend_gan.train()
    L_l2, L_adv = losses
    loss_per_epoch = {'blend_gan': [],
                    'discriminator': []}  # recording losses to plot later

    save_samples = save_sample_single if cfg.LOG.SAVE_SINGLES else save_sample_sheet

    pbar = tqdm(range(*ep_range))

    for i, ep in enumerate(pbar):

        """ Training the blend_gan """
        opts.zero_grad(['blend_gan'])

        # generate x_gt (from cGAN)
        x_gt, mask, premask, foreground, background, background_mask = cgn()

        # generate x (copy + paste composition)
        x = mask * foreground + (1 - mask) * background

        # get the low resolution, well-blended, semantic & colour accurate output x_l
        x_l = blend_gan(x)
        
        # adverserial gts, valid == generated from the blend gan
        valid = torch.ones(x_gt.size(0),).to(device)  # generate labels of length batch_size
        fake = torch.zeros(x_gt.size(0),).to(device) 

        validity = discriminator(x_l)
        # calculate losses
        losses_g = {} 
        losses_g['L_l2'] = L_l2(x_l, x_gt)
        losses_g['L_adv'] = L_adv(validity, valid)
        # print(f"LOSSES in epi: {ep}", losses_g)

        loss_g = sum(losses_g.values())
        loss_g.backward()
        opts.step(['blend_gan'], False)

        # record average loss per batch
        loss_per_epoch['blend_gan'].append(int((loss_g/cfg.TRAIN.BATCH_ACC).item()))
        
        """ Training the discriminator """ 
        opts.zero_grad(['discriminator'])

        #Discriminate real and fake
        validity_real = discriminator(x_gt)
        validity_fake = discriminator(x_l.detach())

        # Losses
        losses_d = {}
        losses_d['real'] = L_adv(validity_real, valid)
        losses_d['fake'] = L_adv(validity_fake, fake)
        loss_d = sum(losses_d.values()) / 2
        # print(f"DISC LOSSES in epi: {ep}", losses_d)

        # Backprop and step
        loss_d.backward()
        opts.step(['discriminator'], False)

        # record average loss per batch for the discriminator
        loss_per_epoch['discriminator'].append(int((loss_g/cfg.TRAIN.BATCH_ACC).item()))
        

        # Saving
        if not i % cfg.LOG.SAVE_ITER:
            ep_str = f'ep_{ep:07}'
            save_samples(blend_gan, cgn, sample_path, ep_str)
            torch.save(blend_gan.state_dict(), join(weights_path, ep_str + '.pth'))

        # Logging
        if cfg.LOG.LOSSES:  
            msg = ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
            pbar.set_description(msg)
            save_samples(blend_gan, cgn, sample_path, ep_str)
            torch.save(blend_gan.state_dict(), join(weights_path, ep_str + '.pth'))

    if cfg.LOG.LOSSES: # TODO: NOT WORKING
        path = join(loss_path, 'losses.csv')
        with open(path, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(loss_per_epoch.keys())
            writer.writerows(zip(*loss_per_epoch.values()))

    
def main(cfg):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # init model
    blend_gan = BlendGAN()
    # init discriminator
    discriminator = Encoder(as_discriminator=True)

    # init cgn
    cgn = CGN(
        batch_sz=cfg.TRAIN.BATCH_SZ,
        truncation=cfg.MODEL.TRUNCATION,
        pretrained=True,
    )

    if cfg.CGN_WEIGHTS_PATH:
        # print(f"Loading CGN weights from {cfg.CGN_WEIGHTS_PATH}")
        weights = torch.load(cfg.CGN_WEIGHTS_PATH, map_location=torch.device(device))
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        cgn.load_state_dict(weights)
    
    # optimizers
    opts = Optimizers()
    opts.set('blend_gan', blend_gan, lr=cfg.LR.BGAN)
    opts.set('discriminator', discriminator, lr=cfg.LR.DISC)
    
    #losses
    L_l2 = ReconstructionLoss(mode='l2', loss_weight=cfg.LAMBDA.L2)
    L_adv = torch.nn.MSELoss()  # ToDo: add correct loss
    losses = (L_l2, L_adv)

    # push to device and train
    blend_gan = blend_gan.to(device)
    discriminator = discriminator.to(device)

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
    parser.add_argument('--batch_sz', type=int, default=4000,
                        help='Batch size, use in conjunciton with batch_acc')
    parser.add_argument('--batch_acc', type=int, default=4000,
                        help='pseudo_batch_size = batch_acc*batch size')

    # ToDo: add more
    args = parser.parse_args()

    cfg = get_cfg_gp_gan_defaults()
    cfg = merge_args_and_cfg(args, cfg)

    print(cfg)
    main(cfg)