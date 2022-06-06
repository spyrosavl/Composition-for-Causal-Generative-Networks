import readline

from experiment_utils import set_env, dotdict
if __name__ == "__main__":
    set_env()

from inception_score_pytorch.inception_score import inception_score
from experiment_utils import ImageDirectoryLoader

from torch.utils.data import Dataset, DataLoader, TensorDataset

import argparse
import torch
import os


def generate_images(final_dir_name, weights_path, run_name):
    '''
    Use the CGN to generate images on which the inception score and mu_mask will be calculated.
    '''
    from cgn_framework.imagenet.generate_data_gp_gan import main as generate_main

    args = dotdict({
        "mode": "random_same",
        # "weights_path": weights_path,
        # "ignore_time_in_filename": True,
        "n_data": 2000,
        "run_name": run_name,
        "truncation": 0.5,
        "batch_sz": 1,
    })

    if os.path.exists(final_dir_name):
        print("Generated data already exists. It will be used instead of regenerated.")
    else:
        print("Genrating new, non-counterfactual data...")
        generate_main(args)

    return final_dir_name

def mu_mask(file_path):
    '''
    Reads the mu_mask values from the text file at location `file_path` (one float per line).
    Then calculates their mean and standard deviation.
    '''
    with open(file_path, 'r') as f:
        mus = f.readlines()
        mus_count = len(mus)

    mus = [float(mu[:-1]) for mu in mus]
    avg_mask = sum(mus) / mus_count
    sds_mask = ((mu - avg_mask) ** 2 for mu in mus)
    sd_mask = (sum(sds_mask) / mus_count) ** 0.5
    return avg_mask, sd_mask

def main(args):
    if not os.path.exists(args.data_dir):
        assert args.run_name is not None and args.weights_path is not None, "if the data_dir argument is not supplied, supply run_name and weight_path"
        data_dir = generate_images(args.data_dir, args.weights_path, args.run_name)
    else:
        data_dir = args.data_dir

    images = ImageDirectoryLoader(data_dir + '/ims')
    inception = inception_score(images, splits=2, resize=True)

    print('inception score =', inception)

    print('mu_mask, sd_mask =', mu_mask(data_dir + '/mean_masks.txt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()                  #ToDo:
    parser.add_argument('--data_dir', type=str, default= '/home/lcur1339/dl2-cgn/cgn_framework/imagenet/data/cgnxgp/GP_refinement/2022_06_06_14_for_inception_trunc_0.5', #"/home/lcur1339/dl2-cgn/cgn_framework/imagenet/data/cgnxgp/GP_refinement/2022_06_06_10_for_inception_trunc_1.0",
                        help='Folder to load the images from. If the images '
                        'do not exist in this folder, the program genetates '
                        'new images and stores those in this folder.')
    parser.add_argument('--weights_path', type=str, default=None)
    parser.add_argument('--run_name', type=str, default=None)

    args = parser.parse_args()

    main(args)