#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=medium
#SBATCH --time=02-00:00:00
#SBATCH --output=TrainCracksComb3.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/TreeRingComb3.py train --dataset=/groups/swarts/user/miroslav.polacek/CNN/treeringCombined  --weights=/users/miroslav.polacek/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb320210305T1124/mask_rcnn_treeringcrackscomb3_0394.h5
