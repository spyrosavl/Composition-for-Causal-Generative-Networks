import experiment_utils
experiment_utils.set_env()

from experiment_utils import load_generated_imagenet
from inception_score_tf import get_inception_score

import argparse
import os
import tensorflow as tf

from matplotlib import pyplot as plt


def generate_images():
    raise NotImplementedError

def load_generated_imagenet(images_dir, images_count=None):
    # Get the locations of the generated images
    image_paths = (images_dir + "/" + path for path in os.listdir(images_dir))

    return [plt.imread(path) for path in image_paths]

def main(args):
    if not os.path.exists(args.image_dir):
        generate_images()

    images = load_generated_imagenet(args.image_dir, args.images_count)
    # print(inception_score(torch.randn((64, 3, 256, 256))))
    inception = get_inception_score([image for image in images])

    print(inception)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True,
                        help='Folder to load the images from. If the images '
                        'do not exist in this folder, the program genetates '
                        'new images and stores those in this folder.')

    parser.add_argument('--images_count', default=None, type=int,
                        help='Number of images to load')
    args = parser.parse_args()

    main(args)
