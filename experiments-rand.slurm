#!/bin/bash
#SBATCH --job-name=cifar_rand_post100_comp025
#SBATCH -N 1
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --time=01:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=mn3265@cs.princeton.edu
#SBATCH --output=/n/fs/mn3265cos568/COS568/torch_demo/out/rand.out
#SBATCH --error=/n/fs/mn3265cos568/COS568/torch_demo/out/rand.err


echo 'Your job is running on node(s):'
echo $SLURM_NODELIST

source /n/fs/mn3265cos568/COS568/torch_demo/venv/bin/activate

python /n/fs/mn3265cos568/COS568/torch_demo/COS568-A1/main.py \
--model resnet20 \
--model-class lottery \
--dataset cifar10 \
--experiment singleshot \
--pruner rand \
--compression 0.25 \
--post-epochs 100 \
--expid cifar_rand_post100_comp025