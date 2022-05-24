import os

from scipy.stats import truncnorm
import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import cv2 as cv
from imagenet.models import BigGAN, U2NET
from utils import toggle_grad

import repackage
repackage.up(1)
from imagenet.dataloader import get_imagenet_dls
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def get_all_patches(ims, patch_sz=[5, 5], pad=True):
    '''
    given a batch of images, get the surrounding patch (of patch_sz=(height,width)) for each pixel
    '''
    assert isinstance(patch_sz, list) and len(patch_sz) == 2, "Wrong format for patch_sz"

    # Pad the images - we want the patch surround each pixel
    patch_sz = np.array(patch_sz)
    patch_sz += (patch_sz+1) % 2  # round up to odd number

    # padding if we want to get the surrounding patches for *all* pixels
    if pad:
        pad = tuple((patch_sz//2).repeat(2))
        ims = nn.ReflectionPad2d(pad)(ims)

    # unfold the last 2 dimensions to get all patches
    patches = ims.unfold(2, patch_sz[0], 1).unfold(3, patch_sz[1], 1)

    # reshape to no_pixel x c x patch_sz x patch_sz
    batch_sz, c, w, h = patches.shape[:4]
    patch_batch = patches.reshape(batch_sz, c, w*h, patch_sz[0], patch_sz[1])
    patch_batch = patch_batch.permute(0, 2, 1, 3, 4)
    patch_batch = patch_batch.reshape(batch_sz*w*h, c, patch_sz[0], patch_sz[1])

    if pad: assert patch_batch.shape[0] == batch_sz * w * h  # one patch per pixel per image

    return patch_batch

def get_sampled_patches(prob_maps, paint, patch_sz=[15, 15], sample_sz=400, n_up=None):
    paint_shape = paint.shape[-2:]
    prob_maps = F.interpolate(prob_maps, (128, 128), mode='bicubic', align_corners=False)
    paint = F.interpolate(paint, (128, 128), mode='bicubic', align_corners=False)

    mode_patches = []
    if n_up is None:
        n_up = paint.shape[-1]//patch_sz[0]

    for p, prob in zip(paint, prob_maps):
        prob_patches = get_all_patches(prob.unsqueeze(0), patch_sz, pad=False)
        prob_patches_mean = prob_patches.mean((1, 2, 3))
        max_ind = torch.argsort(prob_patches_mean)[:sample_sz]  # get 400 topvalues
        max_ind = max_ind[torch.randint(len(max_ind), (n_up**2,))].squeeze()  # sample one
        p_patches = get_all_patches(p[None], patch_sz, pad=False)
        patches = p_patches[max_ind]
        patches = make_grid(patches, nrow=n_up, padding=0)
        patches = F.interpolate(patches[None], paint_shape, mode='bicubic', align_corners=False)
        mode_patches.append(patches)

    return torch.cat(mode_patches)
class CGN():
    def __init__(self, *args, **kwargs):
        self.us2net = U2NET.initialize('../cgn_framework/imagenet/weights/u2net.pth').eval()
        self.train_loader, _, _ = get_imagenet_dls(root="../cgn_framework/imagenet/data/in-mini/", distributed=False, batch_size=32, workers=4)
        self.device = torch.device('cpu')
        pass
    
    def eval(self):
        return self

    def to(self, device):
        self.device = device
        self.us2net.to(device)
        return self

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, ys, debug=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        assert len(ys) == 3, 'Provide 3 classes'
        shape_class, fg_class, bg_class = ys[0], ys[1], ys[2]

        shape_idx = np.random.choice((self.train_loader.dataset.labels == shape_class).nonzero()[0])
        fg_idx = np.random.choice((self.train_loader.dataset.labels == fg_class).nonzero()[0])
        bg_idx = np.random.choice((self.train_loader.dataset.labels == bg_class).nonzero()[0])

        #Images
        shape_img = self.train_loader.dataset.load_image(shape_idx)
        fg_img = self.train_loader.dataset.load_image(fg_idx)
        bg_img = self.train_loader.dataset.load_image(bg_idx)
        
        if debug:
            #Plot images
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].imshow(shape_img.squeeze(0).numpy().transpose(1,2,0))
            ax[1].imshow(fg_img.squeeze(0).numpy().transpose(1,2,0))
            ax[2].imshow(bg_img.squeeze(0).numpy().transpose(1,2,0))
            plt.show()

        #Find Masks
        threshold = 0.2

        batch_imgs = torch.stack([shape_img, fg_img, bg_img]).to(self.device)

        masks = self.us2net(batch_imgs).squeeze(1).unsqueeze(3).detach()

        if debug:
            #Plot masks
            fig, ax = plt.subplots(1,3, figsize=(15,5))
            ax[0].imshow(masks[0].cpu().numpy())
            ax[1].imshow(masks[1].cpu().numpy())
            ax[2].imshow(masks[2].cpu().numpy())
            plt.show()

        shape_mask = (masks[0] > threshold).float()
        fg_mask = (masks[1] > threshold).float()
        bg_mask = (masks[2] > threshold).float()


        #Inpaint bg (fill bg holes)
        bg_img_cropped = np.clip((1-bg_mask) * bg_img.transpose(0,1).transpose(1,2), 0, 1)
        bg_img_cv = (bg_img_cropped.numpy()*255).astype(np.uint8)
        bg_img_inpainted = cv.inpaint(bg_img_cv, bg_mask.numpy().astype(np.uint8), 5, cv.INPAINT_NS)/255

        texture = get_sampled_patches((1-fg_mask).unsqueeze(0).transpose(1,3), fg_img.unsqueeze(0))

        mask = shape_mask.unsqueeze(0).transpose(2,3).transpose(1,2)
        bg_img_inpainted = torch.Tensor(bg_img_inpainted.transpose(2,0,1)).unsqueeze(0)

        return None, mask, None, texture, bg_img_inpainted, None
