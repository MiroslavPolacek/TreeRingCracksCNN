#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --time=0-08:00:00
#SBATCH --output=ExampleDetections.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN



~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/example_detections.py  --weight=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/mask_rcnn_treeringcrackscomb_0284.h5
