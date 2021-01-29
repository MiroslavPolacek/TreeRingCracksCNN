#!/usr/bin/env bash

#SBATCH --nodes=4
#SBATCH --array=0-3
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=2-00:00:00
#SBATCH --output=eval_mAP_parallel_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(treeringcrackscomb220210114T1540 treeringcrackscomb2_equalloss20210115T0012 treeringcrackscomb2_onlycracks20210121T2224 treeringcrackscomb2_onlyring20210121T1457)

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/eval_epochs_mAP_new.py --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new  --weight_folder=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/${FILE[$SLURM_ARRAY_TASK_ID]}
