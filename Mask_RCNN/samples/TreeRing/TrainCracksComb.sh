#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=long
#SBATCH --time=8-00:00:00
#SBATCH --output=TrainCracksComb.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/TreeRingComb.py train --dataset=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/treeringCombined  --weights=last
