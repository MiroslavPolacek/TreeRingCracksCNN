#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=eval_weight.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN



time ~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights.py --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new  --weight=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackstrans220201201T1850/mask_rcnn_treeringcrackstrans2_0040.h5
