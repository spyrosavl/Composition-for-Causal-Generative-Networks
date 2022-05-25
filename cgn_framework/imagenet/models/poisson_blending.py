# Original code from https://github.com/parosky/poissonblending
# Python code from https://github.com/ChengBinJin/semantic-image-inpainting

import numpy as np
import scipy.sparse
from scipy.sparse import linalg
import pyamg
import torch

# pre-process the mask array so that uint64 types from opencv.imread can be adapted
def prepare_mask(mask):
    if type(mask[0][0]) is np.ndarray:
        result = np.ndarray((mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if sum(mask[i][j]) > 0:
                    result[i][j] = 1
                else:
                    result[i][j] = 0
        mask = result
    return mask


def normalize(tensor):
    return (tensor + tensor.min().abs()) / (tensor.max() - tensor.min())

def poissonSeamlessCloning(img_source, img_target, src_mask, offset=(0, 0)): 
    assert img_source.shape == img_target.shape
    assert img_source.ndim == 3 and img_target.ndim == 3 and src_mask.ndim == 3
    assert img_source.shape[2] == img_target.shape[2] and img_source.shape[2] == 3
    assert src_mask.dtype == torch.uint8 and img_source.dtype == torch.float32 and img_target.dtype == torch.float32
    assert src_mask.all() >= 0 and src_mask.all() <= 1

    img_source = img_source.to(torch.float32).numpy().astype(np.float32)
    img_target = img_target.to(torch.float32).numpy().astype(np.float32)
    src_mask = src_mask.numpy()

    assert img_source.max() <= 1 and img_source.min() >= -1
    assert img_target.max() <= 1 and img_target.min() >= -1

    # compute regions to be blended
    region_source = (max(-offset[0], 0), max(-offset[1], 0),
                     min(img_target.shape[0] - offset[0], img_source.shape[0]),
                     min(img_target.shape[1] - offset[1], img_source.shape[1]))
    region_target = (max(offset[0], 0), max(offset[1], 0),
                     min(img_target.shape[0], img_source.shape[0] + offset[0]),
                     min(img_target.shape[1], img_source.shape[1] + offset[1]))
    region_size = (region_source[2] - region_source[0], region_source[3] - region_source[1])

    # clip and normalize mask image
    src_mask = src_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    src_mask = prepare_mask(src_mask)
    src_mask[src_mask == 0] = False
    # img_mask[img_mask != False] = True
    src_mask[src_mask != 0] = True

    # create coefficient matrix
    # a_ = scipy.sparse.identity(np.prod(region_size), format='lil')
    a_ = scipy.sparse.identity(int(np.prod(region_size)), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if src_mask[y, x]:
                index = x + y * region_size[1]
                a_[index, index] = 4
                if index + 1 < np.prod(region_size):
                    a_[index, index + 1] = -1
                if index - 1 >= 0:
                    a_[index, index - 1] = -1
                if index + region_size[1] < np.prod(region_size):
                    a_[index, index + region_size[1]] = -1
                if index - region_size[1] >= 0:
                    a_[index, index - region_size[1]] = -1    

    a_ = a_.tocsr()

    # create poisson matrix for b
    p_ = pyamg.gallery.poisson(src_mask.shape)

    output = img_target.copy()

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[2]):
        # get subimages
        t = img_target[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer]
        s = img_source[region_source[0]:region_source[2], region_source[1]:region_source[3], num_layer]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = p_ * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not src_mask[y, x]:
                    index = x + y * region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        #x = pyamg.solve(a_, b, verb=False, tol=1e-10)
        x = torch.tensor(linalg.spsolve(a_, b))
        # assign x to target image
        x = np.reshape(x, region_size)
        x = np.clip(x, -1, 1)
        x = np.array(x, img_target.dtype)
        output[region_target[0]:region_target[2], region_target[1]:region_target[3], num_layer] = x

    return output
