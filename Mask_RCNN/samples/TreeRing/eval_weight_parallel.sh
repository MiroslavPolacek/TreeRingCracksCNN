#!/usr/bin/env bash

#SBATCH --nodes=2
#SBATCH --array=0-1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=eval_weight_parallel_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(mask_rcnn_treeringcrackscomb_0284.h5 mask_rcnn_treeringcrackscomb_0257.h5)


time ~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights.py --dataset=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/Old_val  --weight=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/${FILE[$SLURM_ARRAY_TASK_ID]}
