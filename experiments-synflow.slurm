#!/bin/bash
#SBATCH --job-name=mnist_synflow_comp1_post100
#SBATCH -N 1
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --time=02:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=mn3265@cs.princeton.edu
#SBATCH --output=/n/fs/mn3265cos568/COS568/torch_demo/out/synflow.out
#SBATCH --error=/n/fs/mn3265cos568/COS568/torch_demo/out/synflow.err


echo 'Your job is running on node(s):'
echo $SLURM_NODELIST

source /n/fs/mn3265cos568/COS568/torch_demo/venv/bin/activate

python /n/fs/mn3265cos568/COS568/torch_demo/COS568-A1/main.py \
--model fc \
--model-class default \
--dataset mnist \
--experiment singleshot \
--pruner synflow \
--compression 1 \
--post-epochs 100 \
--expid mnist_synflow_comp1_post100