#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=long
#SBATCH --time=8-00:00:00
#SBATCH --output=TrainCracksComb2.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/TreeRingComb2.py train --dataset=/groups/swarts/user/miroslav.polacek/CNN/treeringCombined  --weights=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb220210114T1540/mask_rcnn_treeringcrackscomb2_0228.h5
