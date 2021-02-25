#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=64G
#SBATCH --qos=long
#SBATCH --time=14-00:00:00
#SBATCH --output=TrainCracksComb3_onlyCracks.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/TreeRingComb3_onlyCracks.py train --dataset=/groups/swarts/user/miroslav.polacek/CNN/treeringCombined  --weights=imagenet
