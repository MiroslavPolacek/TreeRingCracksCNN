#!/usr/bin/env bash

#SBATCH --nodes=5
#SBATCH --array=0-4
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=eval_weight_comb2_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(TreeRingComb2 TreeRingComb2_95 TreeRingComb2_98 TreeRingComb2_99 TreeRingComb2_100)

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights_conf.py --TreeRingConf=${FILE[$SLURM_ARRAY_TASK_ID]} --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb220210114T1540/mask_rcnn_treeringcrackscomb2_0196.h5
