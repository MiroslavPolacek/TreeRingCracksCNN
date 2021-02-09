#!/usr/bin/env bash

#SBATCH --nodes=2
#SBATCH --array=0-1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=eval_weight_parallel_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(treeringcrackscomb2_equalloss20210115T0012/mask_rcnn_treeringcrackscomb2_equalloss_0185.h5 treeringcrackscomb220210114T1540/mask_rcnn_treeringcrackscomb2_0196.h5)

time ~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights.py --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/${FILE[$SLURM_ARRAY_TASK_ID]}
