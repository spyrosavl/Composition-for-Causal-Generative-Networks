""" implementing an output discriminator for the imagenet CGN"""
import os
from datetime import datetime
from os.path import join
import pathlib
from tqdm import tqdm
import argparse

import torch
from torch import nn, optim
from torch.autograd import Variable
import torchvision
from torchvision.transforms import Pad
from torchvision.utils import make_grid
import repackage
repackage.up()

from imagenet.models import CGN, DiscLin
from imagenet.models.cgn_for_disc import CGNfDISC
from imagenet.config import get_cfg_defaults
from imagenet.config_disc import get_cfg_disc_defaults
from shared.losses import *
from utils import Optimizers

def save_sample_sheet(cgn, u_fixed, sample_path, ep_str):
    cgn.eval()
    dev = u_fixed.to(cgn.get_device())
    ys = [15, 251, 330, 382, 385, 483, 559, 751, 938, 947, 999]

    to_save = []
    with torch.no_grad():
        for y in ys:
            # generate
            y_vec = cgn.get_class_vec(y, sz=1)
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
    cgn.train()

def save_sample_single(cgn, u_fixed, sample_path, ep_str):
    cgn.eval()
    dev = u_fixed.to(cgn.get_device())

    ys = [15, 251, 330, 382, 385, 483, 559, 751, 938, 947, 999]
    with torch.no_grad():
        for y in ys:
            # generate
            y_vec = cgn.get_class_vec(y, sz=1)
            inp = (u_fixed.to(dev), y_vec.to(dev), cgn.truncation)
            _, mask, premask, foreground, background, _ = cgn(inp)
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

    cgn.train()

def fit(cfg, cgn, discriminator, opts, losses, device):

    # total number of episodes, accounted for batch accumulation
    episodes = cfg.TRAIN.EPISODES
    episodes *= cfg.TRAIN.BATCH_ACC

    # directories for experiments
    time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    if cfg.WEIGHTS_PATH:
        weights_path = str(pathlib.Path(cfg.WEIGHTS_PATH).parent)
        start_ep = int(pathlib.Path(cfg.WEIGHTS_PATH).stem[3:])
        sample_path = weights_path.replace('weights', 'samples')
        ep_range = (start_ep, start_ep + episodes)
    else:
        model_path = join('imagenet', 'experiments',
                          f'cgn_{time_str}_{cfg.MODEL_NAME}')
        weights_path = join(model_path, 'weights')
        sample_path = join(model_path, 'samples')
        pathlib.Path(weights_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(sample_path).mkdir(parents=True, exist_ok=True)
        ep_range = (0, episodes)

    # fixed noise sample
    u_fixed_path = join('imagenet', 'experiments', 'u_fixed.pt')
    if not os.path.isfile(u_fixed_path) or cfg.LOG.SAMPLED_FIXED_NOISE:
        u_fixed = cgn.get_noise_vec()
        torch.save(u_fixed, u_fixed_path)
    else:
        u_fixed = torch.load(u_fixed_path)

    # Training Loop
    cgn.train()
    L_l1, L_perc, L_binary, L_mask, L_text, L_bg, L_adv= losses
    save_samples = save_sample_single if cfg.LOG.SAVE_SINGLES else save_sample_sheet

    pbar = tqdm(range(*ep_range))
    for i, ep in enumerate(pbar):
        
        #
        # Training CGN & cGAN
        #
        x_gt, mask, premask, foreground, background, background_mask, inp = cgn()
        x_gen = mask * foreground + (1 - mask) * background
        
        y_gen = inp[1].to(device).long().squeeze(-1)  # random choice of class, same for all IMs; getting the one-hot encoded class
        print(f"Ygen is shape {y_gen.shape}, len {len(y_gen)}, and looks like: {y_gen} ")
        valid = torch.ones(len(y_gen),).to(device)  
        fake = torch.zeros(len(y_gen),).to(device)
        
        # Calc Losses                  
        validity = discriminator(x_gen, y_gen)  # binary vector of len=batch_size (1=fake/generated)

        # Losses
        losses_g = {}
        losses_g['adv'] = L_adv(validity, valid)  # how many the generated ims did the discriminator get right
        losses_g['l1'] = L_l1(x_gen, x_gt)
        losses_g['perc'] = L_perc(x_gen, x_gt)
        losses_g['binary'] = L_binary(mask)
        losses_g['mask'] = L_mask(mask)
        losses_g['perc_text'] = L_text(x_gt, mask, foreground)
        losses_g['bg'] = L_bg(background_mask)

        # backprop
        losses_g = {k: v.mean() for k, v in losses_g.items()}
        g_loss = sum(losses_g.values())
        g_loss.backward()

        if (i+1) % cfg.TRAIN.BATCH_ACC == 0:
            opts.step(['shape', 'bg', 'texture'])

        # Saving
        if not i % cfg.LOG.SAVE_ITER:
            ep_str = f'ep_{ep:07}'
            save_samples(cgn, u_fixed, sample_path, ep_str)
            torch.save(cgn.state_dict(), join(weights_path, ep_str + '.pth'))

        # Logging
        if cfg.LOG.LOSSES:
            msg = ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
            pbar.set_description(msg)

        #
        # Train Discriminator
        #
        opts.zero_grad(['discriminator'])  # ToDo: add discriminator and its optimizer in main

        # Discriminate real and fake
        validity_real = discriminator(x_gt, y_gen[0])
        validity_fake = discriminator(x_gen.detach(), y_gen[0])

        # Discriminator losses
        losses_d = {}
        losses_d['real'] = L_adv(validity_real, valid)
        losses_d['fake'] = L_adv(validity_fake, fake)
        loss_d = sum(losses_d.value()) / 2

        # Backprop and gd step
        loss_d.backward()
        opts.step(['discriminator'], False)

        #  # Saving
        # batches_done = epoch * len(dataloader) + i
        # if batches_done % cfg.LOG.SAVE_ITER == 0:
        #     print("Saving samples and weights")
        #     sample_image(cgn, sample_path, batches_done, device, n_row=3)
        #     torch.save(cgn.state_dict(), f"{weights_path}/ckp_{batches_done:d}.pth")

        # # Logging
        # if cfg.LOG.LOSSES:
        #     msg = f"[Batch {i}/{len(dataloader)}]"
        #     msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_d.items()])
        #     msg += ''.join([f"[{k}: {v:.3f}]" for k, v in losses_g.items()])
        #     pbar.set_description(msg)

def main(cfg):
    # model init
    cgn = CGNfDISC(
        batch_sz=cfg.TRAIN.BATCH_SZ,
        truncation=cfg.MODEL.TRUNCATION,
        pretrained=True,
    )
    Discriminator = DiscLin  #if cfg.MODEL.DISC == 'linear' else DiscConv
    discriminator = Discriminator(n_classes=cfg.MODEL.N_CLASSES, ndf=cfg.MODEL.NDF)
    

    if cfg.WEIGHTS_PATH:
        weights = torch.load(cfg.WEIGHTS_PATH)
        weights = {k.replace('module.', ''): v for k, v in weights.items()}
        cgn.load_state_dict(weights)

    # optimizers
    opts = Optimizers()
    opts.set('shape', cgn.f_shape, cfg.LR.SHAPE)
    opts.set('texture', cgn.f_text, cfg.LR.TEXTURE)
    opts.set('bg', cgn.f_bg, cfg.LR.BG)
    
    opts.set('discriminator', discriminator, lr=cfg.LR.LR, betas=cfg.LR.BETAS)  # ToDo: check if okay

    # losses
    L_adv = torch.nn.MSELoss()  # adversarial module loss
    L_l1 = ReconstructionLoss(mode='l1', loss_weight=cfg.LAMBDA.L1)
    L_perc = PerceptualLoss(style_wgts=cfg.LAMBDA.PERC)
    L_binary = BinaryLoss(loss_weight=cfg.LAMBDA.BINARY)
    L_mask = MaskLoss(loss_weight=cfg.LAMBDA.MASK)
    L_text = PercLossText(style_wgts=cfg.LAMBDA.TEXT)
    L_bg = BackgroundLoss(loss_weight=cfg.LAMBDA.BG)
    losses = (L_l1, L_perc, L_binary, L_mask, L_text, L_bg, L_adv)

    # push to device and train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cgn = cgn.to(device)
    losses = (l.to(device) for l in losses)

    fit(cfg, cgn, discriminator, opts, losses, device)

def merge_args_and_cfg(args, cfg):
    cfg.MODEL_NAME = args.model_name
    cfg.WEIGHTS_PATH = args.weights_path

    cfg.LOG.SAMPLED_FIXED_NOISE = args.sampled_fixed_noise
    cfg.LOG.SAVE_SINGLES = args.save_singles
    cfg.LOG.SAVE_ITER = args.save_iter
    cfg.LOG.LOSSES = args.log_losses

    cfg.TRAIN.EPISODES = args.episodes
    cfg.TRAIN.BATCH_SZ = args.batch_sz
    cfg.TRAIN.BATCH_ACC = args.batch_acc

    cfg.MODEL.TRUNCATION = args.truncation
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='tmp',
                        help='Weights and samples will be saved under experiments/model_name')
    parser.add_argument('--weights_path', default='',
                        help='provide path to continue training')
    parser.add_argument('--sampled_fixed_noise', default=False, action='store_true',
                        help='If you want a different noise vector than provided in the repo')
    parser.add_argument('--save_singles', default=False, action='store_true',
                        help='Save single images instead of sheets')
    parser.add_argument('--truncation', type=float, default=1.0,
                        help='Truncation value for noise sampling')
    parser.add_argument('--episodes', type=int, default=300,
                        help="We don't do dataloading, hence, one episode = one gradient update.")
    parser.add_argument('--batch_sz', type=int, default=1,
                        help='Batch size, use in conjunciton with batch_acc')
    parser.add_argument('--batch_acc', type=int, default=4000,
                        help='pseudo_batch_size = batch_acc*batch size')
    parser.add_argument('--save_iter', type=int, default=4000,
                        help='Save samples/weights every n iter')
    parser.add_argument('--log_losses', default=False, action='store_true',
                        help='Print out losses')
    args = parser.parse_args()

    cfg = get_cfg_disc_defaults()
    cfg = merge_args_and_cfg(args, cfg)

    print("Num episodes", args.episodes)
    print("Batch acc", args.batch_acc)

    print(cfg)
    main(cfg)
