""" Implementing a Gaussian-Poisson GAN for image plending 
Inspired from: https://arxiv.org/pdf/1703.07195.pdf - https://github.com/wuhuikai/GP-GAN/blob/master/gp_gan.py

Two pipelines: 1) capturing a low-resolution, but accurate colour image C(x)(as the Auto-encoder G(x)) and 2) the gradients (texture) of the image P(x)
    - The Gaussian-Poisson equation is optimised s.t. C(x) and P(x). Then upsample. The optimize the G-P equation at the higher scale to get final result
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
import numpy as np
import cv2

# custom weights initialization 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight)

class Encoder(nn.Module):
    
    def __init__(self, ngf=64, z_size=4000, as_discriminator=False, img_size = (3, 256, 256)):
        super(Encoder, self).__init__()

        self.discriminator = as_discriminator

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=(4,4), stride=2, padding=1, bias=False), # outputs 112x112
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2),  
        )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=z_size, kernel_size=(4,4), stride=1, padding=0, bias=False)  
        )  # outputs (Nx4000x13x13)

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(131072, 1)
        )

        self.model.apply(weights_init)
        self.output.apply(weights_init)
        self.clf.apply(weights_init)

    def forward(self, input):
        x = self.model(input)
        if self.discriminator:
            x = self.clf(x).squeeze(1)  # returns tensor of size [1]
        else:
            x = self.output(x)
        return x


class Decoder(nn.Module):

    def __init__(self, ndf=512, z_size=4000, n_classes=1000, out_img_size= (3, 256, 256)):
        super(Decoder, self).__init__()

        # self.dconv1 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_size, out_channels=ndf, kernel_size=(4,4), stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.2),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.output = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(3,3), stride=1, padding=1, bias=False),
            nn.Tanh()
        )

        self.model.apply(weights_init)
        self.output.apply(weights_init)

    def forward(self, input):
        x = self.model(input)
        return self.output(x)


class BlendGAN(nn.Module):

    def __init__(self, z_size=4000): #encoder, decoder):
        super(BlendGAN, self).__init__()
        self.encoder = Encoder(z_size=z_size)
        self.bn = nn.BatchNorm2d(z_size)
        self.decoder = Decoder(z_size=z_size)

    def forward(self, input):
        x = self.encoder.model(input)
        x = self.encoder.output(x)
        x = self.bn(x)
        x = self.decoder.model(x)
        x = self.decoder.output(x)
        return x


""" Functions for the G-P bledning """

device =  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def get_img_gradients(img):
#     """ Get the gradients of an image 
#     expects tensor image of shape [CxHxW]
#     outputs image of gradients of shape (Hxw) in nparray format
#     """
#     # Get x-gradient in "sx"
#     sx = ndimage.sobel(img,axis=1,mode='constant')
#     # Get y-gradient in "sy"
#     sy = ndimage.sobel(img,axis=2,mode='constant')
#     # Get square root of sum of squares
#     sobel=np.hypot(sx,sy)
#     return np.mean(sobel, 0)

def get_img_gradients(img):
    """ Get the gradients of an image 
    expects tensor image of shape [HxWxC]
    outputs image of gradients of shape (Hxw) in nparray format
    """
    Gaussian = torch.tensor([[[[1,2,1],[2,4,2],[1,2,1]]]],device=device,dtype=torch.float)
    img_b = torch.stack([torch.nn.functional.conv2d(img[:,:,0].unsqueeze(0).unsqueeze(0), Gaussian,stride=1, padding=1)[0,0]
               ,torch.nn.functional.conv2d(img[:,:,1].unsqueeze(0).unsqueeze(0), Gaussian,stride=1, padding=1)[0,0]
               ,torch.nn.functional.conv2d(img[:,:,2].unsqueeze(0).unsqueeze(0), Gaussian,stride=1, padding=1)[0,0]]).permute(1,2,0)
    return img_b

def imfilter2d(im, filter_func):
    gradients = np.zeros_like(im)
    im_np = im.numpy()
    for i in range(im.shape[2]):
        gradients[:, :, i] = filter_func(im_np[:, :, i])

    return gradients

def gradient_features(img, x_l):

    result = np.zeros((*img.shape, 5))
    #gradient_h, gradient_v = gradient_operator[gradient_kernel]  # ToDo: pass the kernels
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    result[:, :, :, 0] = x_l
    result[:, :, :, 1] = imfilter2d(img, sobely)
    result[:, :, :, 2] = imfilter2d(img, sobelx)
    result[:, :, :, 3] = np.roll(result[:, :, :, 1], 1, axis=1)
    result[:, :, :, 4] = np.roll(result[:, :, :, 2], 1, axis=0)
    return result


def run_gp_editting(foreground, mask, background, xl, beta):
    d_fg = gradient_features(foreground, xl)  # passing the deterministically composed image
    d_bg = gradient_features(background, xl)              
    d_x = mask * d_fg + (1-mask) *  d_bg 

    size, dtype = d_x.shape[:2], d_x.dtype

    return d_x






# #### testing
# import random
# import numpy as np


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import torchvision
# import matplotlib.pyplot as plt

# dtype = torch.double
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# # example python implementation: https://github.com/msinghal34/Image-Blending-using-GP-GANs/blob/master/src/PyramidBlending.py

# class Net(nn.Module):

#     def __init__(self):
#         super(Net, self).__init__()
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#         # Encoder
#         self.batch_norm = nn.BatchNorm2d(3)
#         self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
#         self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
#         self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
#         self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
#         self.conv5 = nn.Conv2d(512, 4000, 4)
        
#         # torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
#         # Staring Decoder
#         self.dconv5 = nn.ConvTranspose2d(4000, 512, 4)
#         self.dconv4 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
#         self.dconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
#         self.dconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
#         self.dconv1 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        
#         torch.nn.init.xavier_normal_(self.conv1.weight)
#         torch.nn.init.xavier_normal_(self.conv2.weight)
#         torch.nn.init.xavier_normal_(self.conv3.weight)
#         torch.nn.init.xavier_normal_(self.conv4.weight)
#         torch.nn.init.xavier_normal_(self.conv5.weight)
        
#         torch.nn.init.xavier_normal_(self.dconv1.weight)
#         torch.nn.init.xavier_normal_(self.dconv2.weight)
#         torch.nn.init.xavier_normal_(self.dconv3.weight)
#         torch.nn.init.xavier_normal_(self.dconv4.weight)
#         torch.nn.init.xavier_normal_(self.dconv5.weight)
        
        

#     def forward(self, x):
#         print("INPUT SIZE", x.size())
#         x1 = F.relu(self.conv1(x))
#         x2 = F.relu(self.conv2(x1))
#         x3 = F.relu(self.conv3(x2)) 
#         x4 = F.relu(self.conv4(x3))
#         x5 = F.relu(self.conv5(x4))
#         print("LATENT VECTOR SIZE", x5.size())
#         x6 = F.relu(self.dconv5(x5))
#         x7 = F.relu(self.dconv4(x6))
#         x8 = F.relu(self.dconv3(x7))
#         x9 = F.relu(self.dconv2(x8))
#         x10 = F.relu(self.dconv1(x9))
#         print("OUTPUT SIZE", x.size())
#         return x10

#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features