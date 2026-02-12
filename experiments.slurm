#!/bin/bash
#SBATCH --job-name=cos568-experiments
#SBATCH -N 1
#SBATCH --gres=gpu:1 --ntasks-per-node=1 -N 1
#SBATCH --time=00:15:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=mn3265@cs.princeton.edu
#SBATCH --output=/n/fs/mn3265cos568/COS568/torch_demo/out/%x_%j.out
#SBATCH --error=/n/fs/mn3265cos568/COS568/torch_demo/out/%x_%j.err


echo 'Your job is running on node(s):'
echo $SLURM_NODELIST

source /n/fs/mn3265cos568/COS568/torch_demo/venv/bin/activate

python /n/fs/mn3265cos568/COS568/torch_demo/COS568-A1/main.py --model-class lottery --model resnet20 --dataset cifar10 --experiment singleshot --pruner rand --compression 1 --post-epoch 100