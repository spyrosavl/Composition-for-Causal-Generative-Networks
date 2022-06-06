
<!-- Template source: https://github.com/paperswithcode/releasing-research-code -->
<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->

# Replication of Counterfactual Generative Networks

This repository is an extension the [replication implementation](https://openreview.net/forum?id=HNlzT3G720t) of [Counterfactual Generative Networks](https://arxiv.org/abs/2030.12345).

# Reproducing our experiments
Please download the datasets and the weights first:

`conda activate cgn-cpu`

`python setup/download_datasets.py`

`python setup/download_weights.py`

Then run the following commands to reproduce the experiments on slurm cluster:

## Fake-CGN baseline

`cd u2net_cropping`

`sbatch run_imagenet.job`

## Poisson blending (Pretrained)

`cd deterministic_refinement`

`sbatch run_imagenet.job`

## GP-GAN

`cd gp_gan_refinement`

`sbatch run_imagenet.job` 

