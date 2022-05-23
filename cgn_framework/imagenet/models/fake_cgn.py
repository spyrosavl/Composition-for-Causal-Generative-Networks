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
repackage.up(2)
from cgn_extensions.imagenet.dataloader import get_imagenet_dls
from torchvision.utils import make_grid


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

def get_sampled_patches(prob_maps, paint, patch_sz=[30, 30], sample_sz=100, n_up=None):
    paint_shape = paint.shape[-2:]
    prob_maps = F.interpolate(prob_maps, (128, 128), mode='bicubic', align_corners=False)
    paint = F.interpolate(paint, (128, 128), mode='bicubic', align_corners=False)

    mode_patches = []
    if n_up is None:
        n_up = paint.shape[-1]//patch_sz[0]

    for p, prob in zip(paint, prob_maps):
        prob_patches = get_all_patches(prob.unsqueeze(0), patch_sz, pad=False)
        prob_patches_mean = prob_patches.mean((1, 2, 3))
        max_ind = torch.argsort(prob_patches_mean)[-sample_sz:]  # get 400 topvalues
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
        pass
    
    def eval(self):
        return self

    def to(self, device):
        self.device = device
        self.us2net.to(device)
        return self

    def load_state_dict(self, state_dict):
        pass

    def __call__(self, ys):
        assert len(ys) == 3, 'Provide 3 classes'
        shape_class, fg_class, bg_class = ys[0], ys[1], ys[2]

        shape_idx = np.random.choice((self.train_loader.dataset.labels == shape_class).nonzero()[0])
        fg_idx = np.random.choice((self.train_loader.dataset.labels == fg_class).nonzero()[0])
        bg_idx = np.random.choice((self.train_loader.dataset.labels == bg_class).nonzero()[0])


        #Shape mask
        shape_img = self.train_loader.dataset.__getitem__(shape_idx)['ims'].unsqueeze(0).to(self.device)
        shape_mask = self.us2net.forward(shape_img).cpu().detach().numpy().squeeze(0)
        shape_mask = (shape_mask > 0.5).astype(np.uint8)
        shape_img = shape_img.cpu().detach().squeeze(0).numpy()

        #Foreground mask
        fg_img = self.train_loader.dataset.__getitem__(fg_idx)['ims'].unsqueeze(0).to(self.device)
        fg_mask = self.us2net.forward(fg_img).cpu().detach().numpy().squeeze(0)
        fg_mask = (fg_mask > 0.5).astype(np.uint8)
        fg_img = fg_img.cpu().detach().squeeze(0).numpy()

        #Background mask
        bg_img = self.train_loader.dataset.__getitem__(bg_idx)['ims'].unsqueeze(0).to(self.device)
        bg_mask = self.us2net.forward(bg_img).cpu().detach().numpy().squeeze(0)
        bg_mask = (bg_mask > 0.5).astype(np.uint8)
        bg_img = bg_img.cpu().detach().squeeze(0).numpy()

        #Inpaint bg
        bg_img_cropped = np.clip((1-bg_mask) * bg_img, 0, 1)
        bg_img_cv = (bg_img_cropped.transpose(1,2,0)*255).astype(np.uint8)
        bg_img_inpainted = cv.inpaint(bg_img_cv, bg_mask.transpose(1,2,0).astype(np.uint8), 5, cv.INPAINT_NS)/255

        texture = get_sampled_patches(torch.Tensor(fg_mask).unsqueeze(0), torch.Tensor(fg_img).unsqueeze(0))
        texture = texture.squeeze(0).numpy().transpose(1,2,0)

        mask = torch.Tensor(shape_mask).unsqueeze(0)
        texture = torch.Tensor(texture.transpose(2,0,1)).unsqueeze(0)
        bg_img_inpainted = torch.Tensor(bg_img_inpainted.transpose(2,0,1)).unsqueeze(0)

        return None, mask, None, texture, bg_img_inpainted, None
