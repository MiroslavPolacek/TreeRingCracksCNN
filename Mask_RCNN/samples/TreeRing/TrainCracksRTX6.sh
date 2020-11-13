#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=TrainCracksRTX6.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/TreeRing6.py train --dataset=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/treering  --weights=BestOnRings
