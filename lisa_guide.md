# Lisa setup
1. Login to lisa with ssh `ssh lcur1334@lisa.surfsara.nl` (change the username)
2. Connect with Github. Run in lisa:
   1.  `ssh-keygen -t ed25519 -C "spyrosavl@gmail.com"` (just press enter for the rest)
   2.  `eval "$(ssh-agent -s)"`
   3.  `ssh-add ~/.ssh/id_ed25519`
   4.  Copy the generated public key from `cat ~/.ssh/id_ed25519.pub` to Github (https://github.com/settings/ssh/new). The name doesn't matter.
3. Clone the project `git clone git@github.com:spyrosavl/dl2-cgn.git`
4. Create the conda environment:
   1. `module load 2021`
   2. `module load Anaconda3/2021.05`
   3. `conda env create -f dl2-cgn/cgn_framework/environment-gpu.yml`
5. Set up Kaggle CLI in order to download datesets
   1. Create a new API token in https://www.kaggle.com/spyrosavl/account
   2. Add `kaggle.json` in `~/.kaggle/kaggle.json` on Lisa.
6. Dowload datasets and weights:
   1. `conda activate dl2-cgn`
   2. `python dl2-cgn/setup/download_datasets.py`
   3. `python dl2-cgn/setup/download_weights.py`

# Run MNIST experiments
1. Submit the job: `cd dl2-cgn/experiments && sbatch run_mnist.job`
2. Check running jobs: `squeue -u lcur1334` (replace with your username)

# Tutorial from the DL1 course
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial1/Lisa_Cluster.html