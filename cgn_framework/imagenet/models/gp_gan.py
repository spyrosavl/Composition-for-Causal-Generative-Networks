""" Implementing a Gaussian-Poisson GAN for image plending 
Inspired from: https://arxiv.org/pdf/1703.07195.pdf - https://github.com/wuhuikai/GP-GAN/blob/master/gp_gan.py

Two pipelines: 1) capturing a low-resolution, but accurate colour image C(x)(as the Auto-encoder G(x)) and 2) the gradients (texture) of the image P(x)
    - The Gaussian-Poisson equation is optimised s.t. C(x) and P(x). Then upsample. The optimize the G-P equation at the higher scale to get final result
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    def __init__(self, z_size=4000, img_size = (3, 256, 256)):
        super(Encoder, self).__init__()

        # self.input_size = input_size
        self.z_sie = z_size

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4), stride=2, padding=1), # outputs 112x112
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),  # Nx512x14x14
        )

    def forward(self, input):
        return self.model(input)



class Decoder(nn.Module):

    def __init__(self, z_size, n_classes=1000, out_img_size= (3, 256, 256)):
        self.z_size = z_size
        self.n_classes=n_classes

    # ToDo: 


class AutoEncoder(nn.Module):

    def __init__(self):
        pass