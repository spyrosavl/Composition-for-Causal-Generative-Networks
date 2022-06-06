
<!-- Template source: https://github.com/paperswithcode/releasing-research-code -->
<!-- >📋  A template README.md for code accompanying a Machine Learning paper -->

# Composition for Causal Generative Networks

This repository is an extension the [replication implementation](https://openreview.net/forum?id=HNlzT3G720t) of [Counterfactual Generative Networks](https://arxiv.org/abs/2030.12345).

# Reproducing our experiments
Please download the datasets and the weights first:

`conda activate cgn-cpu`

`python setup/download_datasets.py`

`python setup/download_weights.py`

Then run the following commands to reproduce the experiments on slurm cluster:
If you want to run the experiments on your local machine, please run the bash code in each of the .job files.


## Fake-CGN baseline

`cd u2net_cropping`

`sbatch run_imagenet.job`

## Poisson blending (Pretrained)

`cd deterministic_refinement`

`sbatch run_imagenet.job`

## Poisson blending (Finetuned)

`cd cgn_framework`

`sbatch run_train_cgn_poisson.job`

`cd deterministic_refinement_retrain`

`sbatch run_imagenet.job`

## GP-GAN

`cd gp_gan_refinement`

`sbatch run_imagenet.job` 

`sbatch run_inception_score.job` 
