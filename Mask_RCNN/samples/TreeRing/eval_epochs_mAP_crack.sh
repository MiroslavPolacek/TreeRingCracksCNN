#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=2-00:00:00
#SBATCH --output=eval_epochs_mAP_crack.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/eval_epochs_mAP_new_crack.py --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new  --weight_folder=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb3_onlycracks20210225T1610
