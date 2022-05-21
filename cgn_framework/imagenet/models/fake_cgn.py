import os

from scipy.stats import truncnorm
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv
from imagenet.models import BigGAN, U2NET
from utils import toggle_grad
from imagenet.dataloader import get_imagenet_dls

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
        shape_mask = self.us2net.forward(shape_img).detach().numpy().squeeze(0)
        shape_mask = (shape_mask > 0.1).astype(np.uint8)
        shape_img = shape_img.squeeze(0).numpy()

        #Foreground mask
        fg_img = self.train_loader.dataset.__getitem__(fg_idx)['ims'].unsqueeze(0).to(self.device)
        fg_mask = self.us2net.forward(fg_img).detach().numpy().squeeze(0)
        fg_mask = (fg_mask > 0.1).astype(np.uint8)
        fg_img = fg_img.squeeze(0).numpy()

        #Background mask
        bg_img = self.train_loader.dataset.__getitem__(bg_idx)['ims'].unsqueeze(0).to(self.device)
        bg_mask = self.us2net.forward(bg_img).detach().numpy().squeeze(0)
        bg_mask = (bg_mask < 0.1).astype(np.uint8)
        bg_img = bg_img.squeeze(0).numpy()

        #Inpaint bg
        bg_img_cropped = np.clip(bg_mask * bg_img, 0, 1)
        bg_img_cv = (bg_img_cropped.transpose(1,2,0)*255).astype(np.uint8)
        bg_img_inpainted = cv.inpaint(bg_img_cv, np.abs(bg_mask-1).transpose(1,2,0).astype(np.uint8), 50, cv.INPAINT_NS)/255

        texture = np.clip((fg_mask * fg_img).transpose(1,2,0), 0, 1)

        mask = torch.Tensor(shape_mask).unsqueeze(0)
        texture = torch.Tensor(texture.transpose(2,0,1)).unsqueeze(0)
        bg_img_inpainted = torch.Tensor(bg_img_cropped).unsqueeze(0)

        return None, mask, None, texture, bg_img_inpainted, None
