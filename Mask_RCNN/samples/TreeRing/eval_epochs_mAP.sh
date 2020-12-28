#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=2-00:00:00
#SBATCH --output=eval_epochs_mAP.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/eval_epochs_mAP.py --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new  --weight_folder=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/
