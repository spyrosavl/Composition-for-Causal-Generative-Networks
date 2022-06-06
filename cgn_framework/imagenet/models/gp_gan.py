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
    
    def __init__(self, ngf=64, z_size=4000, img_size = (3, 256, 256)):
        super(Encoder, self).__init__()


        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=(4,4), stride=2, padding=1, bias=False), # outputs 112x112
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LayerNorm((1, 128,16,16)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            nn.LayerNorm((1, 256, 8, 8)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LayerNorm((1,512, 4, 4)),
            nn.LeakyReLU(negative_slope=0.2),  
        )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=z_size, kernel_size=(4,4), stride=1, padding=0, bias=False)  
        )  # outputs (Nx4000x13x13)

        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1)
        )

        self.model.apply(weights_init)
        self.output.apply(weights_init)
        self.clf.apply(weights_init)

    def forward(self, input):
        x = self.model(input)
        x = self.output(x)
        return x

class Discriminator(nn.Module):
    """ The discriminator, based off the encoder architecture """
    def __init__(self, ngf=64, z_size=4000, img_size = (3, 256, 256)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=ngf, kernel_size=(4,4), stride=2, padding=1, bias=False), # outputs 112x112
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(128),
            nn.LayerNorm((1, 128, 16, 16)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.LayerNorm((1, 256, 8, 8)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(512),
            nn.LayerNorm((1, 512, 4, 4)),
            nn.LeakyReLU(negative_slope=0.2),  
        )
        self.clf = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 1)
        )
        self.model.apply(weights_init)
        self.clf.apply(weights_init)

    def forward(self, input):
        x = self.model(input)
        x = self.clf(x).squeeze(1)  # returns tensor of size [1]
        return x

class Decoder(nn.Module):

    def __init__(self, ndf=512, z_size=4000, n_classes=1000, out_img_size= (3, 256, 256)):
        super(Decoder, self).__init__()

        # self.dconv1 = nn.ConvTranspose2d(64, 3, 4, 2, 1)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=z_size, out_channels=ndf, kernel_size=(4,4), stride=1, padding=0, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(4,4), stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(256),
            nn.LayerNorm((1, 256, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4,4), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(128),
            nn.LayerNorm((1, 128, 16, 16)),
            nn.ReLU(),  
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.LayerNorm((1, 64, 32, 32)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.LayerNorm((1, 32, 64, 64)),
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
        # self.bn = nn.BatchNorm2d(z_size)
        nn.LayerNorm((z_size,32,32)),
        self.decoder = Decoder(z_size=z_size)

    def forward(self, input):
        #print("Input min,max", input.min(), input.max())
        x = self.encoder.model(input)
        #print("PreComp min, max", x.min(), x.max())
        x = self.encoder.output(x)
        #x = self.bn(x)
        #print("Z min,max", x.min(), x.max())
        x = self.decoder.model(x)
        #print("PreOUT min, max", x.min(), x.max())
        x = self.decoder.output(x)
        #print("OUT min,max", x.min(), x.max())
        return x


""" Functions for the G-P image blending pipelie """

device =  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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



import math
from scipy.ndimage import correlate
from scipy.fftpack import dct, idct
from skimage.transform import resize
from skimage.filters import gaussian, sobel_h, sobel_v, scharr_h, scharr_v, roberts_pos_diag, roberts_neg_diag, \
    prewitt_h, prewitt_v

# The G-P algorithm
    # 1. Laplacian pyramid for fg, bg, & mask
    # 2. Get x_l from BlendGAN
    # 3. For every scale in the possible scales S
    # 4.  a. Update xh (at scale s) by optimizing H(xh)
    # 5.  b. Set xl to be upsampled xh (at scale s)
    # 6. 
    # 7. Return xh (at scale S)


# some helper functions
normal_h = lambda im: correlate(im, np.asarray([[0, -1, 1]]), mode='nearest')
normal_v = lambda im: correlate(im, np.asarray([[0, -1, 1]]).T, mode='nearest')
gradient_operator = {
    'normal': (normal_h, normal_v),
    'sobel': (sobel_h, sobel_v),
    'scharr': (scharr_h, scharr_v),
    'roberts': (roberts_pos_diag, roberts_neg_diag),
    'prewitt': (prewitt_h, prewitt_v)
}

def idct2(x, norm='ortho'):
    return idct(idct(x, norm=norm).T, norm=norm).T

def dct2(x, norm='ortho'):
    return dct(dct(x, norm=norm).T, norm=norm).T

def fft2(K, size, dtype):
    w, h = size
    param = np.fft.fft2(K)
    param = np.real(param[0:w, 0:h])

    return param.astype(dtype)

def gaussian_param(size, dtype, sigma):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    K[1, 1] = 1
    K[:3, :3] = gaussian(K[:3, :3], sigma)

    K = np.roll(K, -1, axis=0)
    K = np.roll(K, -1, axis=1)

    return fft2(K, size, dtype)

def laplacian_param(size, dtype):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    laplacian_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = np.roll(K, -(kw // 2), axis=0)
    K = np.roll(K, -(kh // 2), axis=1)

    return fft2(K, size, dtype)

def imfilter2d(im, filter_func):
    gradients = np.zeros_like(im)
    for i in range(im.shape[2]):
        gradients[:, :, i] = filter_func(im[:, :, i])

    return gradients

def ndarray_resize(img, image_size, order=3, dtype=None):
    img = resize(img, image_size, preserve_range=True, order=order, mode='constant')

    if dtype:
        img = img.astype(dtype)
    return img



def gaussian_poisson_editting(X, param_l, param_g, color_weight=1, eps=1e-12):
    Fh = (X[:, :, :, 1] + np.roll(X[:, :, :, 3], -1, axis=1)) / 2
    Fv = (X[:, :, :, 2] + np.roll(X[:, :, :, 4], -1, axis=0)) / 2
    L = np.roll(Fh, 1, axis=1) + np.roll(Fv, 1, axis=0) - Fh - Fv

    param = param_l + color_weight * param_g
    param[(param >= 0) & (param < eps)] = eps
    param[(param<0) & (param > -eps)] = -eps

    Y = np.zeros(X.shape[:3])
    for i in range(3):
        Xdct = dct2(X[:,:,i,0])
        Ydct = (dct2(L[:,:,i]) + color_weight * Xdct) / param
        Y[:,:,i] = idct2(Ydct)
    return Y


# functions used in the gp editting
def gradient_features(img, color_feature, gradient_kernel):
    result = np.zeros((*img.shape, 5))
    gradient_h, gradient_v = gradient_operator[gradient_kernel]  # x: horizontal, y: vertical
    
    result[:, :, :, 0] = color_feature
    result[:, :, :, 1] = imfilter2d(img, gradient_h)
    result[:, :, :, 2] = imfilter2d(img, gradient_v)
    result[:, :, :, 3] = np.roll(result[:, :, :, 1], 1, axis=1)
    result[:, :, :, 4] = np.roll(result[:, :, :, 2], 1, axis=0)

    return result.astype(img.dtype)

# function for computing the gradients 
def run_gp_editting(foreground, background, mask, gan_im, color_weight, sigma, gradient_kernel='normal'):
    bg_features = gradient_features(background, gan_im, gradient_kernel)
    fg_features = gradient_features(foreground, gan_im, gradient_kernel)
    # ToDo: check my assumption here - namely, that I create a feature map on all 5 dimensions
    feature = fg_features * mask[:,:,:, np.newaxis] + bg_features * (1 - mask[:,:,:, np.newaxis])
    #print(feature.shape) #64x64x3x5
    size, dtype = feature.shape[:2], feature.dtype  
    #print(size) #64x64
    #print(dtype) #float32
    param_l = laplacian_param(size, dtype)
    param_g = gaussian_param(size, dtype, sigma)
    gan_im = gaussian_poisson_editting(feature, param_l, param_g, color_weight=color_weight)
    gan_im = np.clip(gan_im, 0, 1)

    #print("GAN IM AFTER GP EDIT", gan_im.shape)
    return gan_im

def laplacian_pyramid(img, max_level, image_size, smooth_sigma):
    """ Each level captures image structure present at a particular scale
        Returns a list of img_pyr and dif_pyr (in our case: (64x64x3, 128x128x3, 256x256x3) 
        & the last two for the dif_pyr)
     """
    img_pyramid = [img]
    diff_pyramid = []
    for i in range(max_level-1, -1, -1):
        smoothed = gaussian(img_pyramid[-1], smooth_sigma, multichannel=True)
        diff_pyramid.append(img_pyramid[-1] - smoothed)
        smoothed = ndarray_resize(smoothed, (image_size*2 ** i, image_size*2 ** i))
        img_pyramid.append(smoothed)
    img_pyramid.reverse()
    diff_pyramid.reverse()

    return img_pyramid, diff_pyramid

def gp_gan(foreground, background, mask, xl, color_weight=1, image_size=256, sigma=0.5, gradient_kernel='normal', smooth_sigma=1, blend_net_configed=False):
    """  The full G-P GAN pipiline (accepts tensors, outputs numpy arrays) """

    # convert to numpy arrays and appropriate dim configuration (apart from the blend_net output)
    fg = foreground.detach().squeeze(0).cpu().numpy().transpose(1,2,0)  # get into (H,W,C) format
    bg = background.detach().squeeze(0).cpu().numpy().transpose(1,2,0) 
    if not blend_net_configed:
        xl = xl.detach().squeeze(0).cpu().numpy().transpose(1,2,0)
    msk = mask.detach().squeeze().cpu().numpy().transpose(1,0)

    W_orig, H_orig, C = fg.shape
    max_level = int(math.ceil(np.log2(max(W_orig, H_orig) / image_size)))
    
    obj_im_pyramid, _ = laplacian_pyramid(fg, max_level, image_size, smooth_sigma)
    bg_im_pyramid, _ = laplacian_pyramid(bg, max_level, image_size, smooth_sigma)

    # get the composite img
    mask_init = ndarray_resize(msk, (image_size, image_size), order=0)[:,:,np.newaxis]
    composite_img = mask_init * obj_im_pyramid[0] + (1-mask_init) * bg_im_pyramid[0]

    gan_im = xl.copy()
    # start the pyramid optimization
    for level in range(max_level+1):
        size = obj_im_pyramid[level].shape[:2]  # get W,H of current level
        msk_im = ndarray_resize(msk, size, order=0)[:,:,np.newaxis]
        #print(msk_im.shape)
        if level != 0:
            gan_im = ndarray_resize(xl, size)
        gan_im = run_gp_editting(obj_im_pyramid[level], bg_im_pyramid[level], msk_im, gan_im, 
                                color_weight, sigma, gradient_kernel)
    
    gan_im = np.clip(gan_im*255, 0, 255).astype(np.uint8)
    
    return gan_im
# example python implementation: https://github.com/msinghal34/Image-Blending-using-GP-GANs/blob/master/src/PyramidBlending.py
