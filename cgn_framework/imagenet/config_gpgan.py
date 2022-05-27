from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

# General
__C.MODEL_NAME = 'tmp'
__C.CGN_WEIGHTS_PATH = ''
__C.BGAN_WEIGHTS_PATH = 'imagenet/experiments/bgn_2022_05_25_20_45_tmp/weights/ep_0023400.pth'

# Logging
__C.LOG = CN()
__C.LOG.SAVE_ITER = 100
__C.LOG.SAMPLED_FIXED_NOISE = False
__C.LOG.SAVE_SINGLES = False
__C.LOG.LOSSES = True

# Model
__C.MODEL = CN()
__C.MODEL.RES = 256
__C.MODEL.TRUNCATION = 1.0

# Training
__C.TRAIN = CN()
__C.TRAIN.EPOCHS = 50
__C.TRAIN.BATCH_SZ = 256
__C.TRAIN.BATCH_ACC = 8

# Loss Weigths
__C.LAMBDA = CN()
__C.LAMBDA.L2 = 0.99
__C.LAMBDA.ADV = 0.11

# Learning Rates
__C.LR = CN()
__C.LR.BGAN = 1e-4
__C.LR.DISC = 1e-4

def get_cfg_gp_gan_defaults():
    return __C.clone()
