#!/usr/bin/env bash

#SBATCH --nodes=5
#SBATCH --array=0-4
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=eval_weight_cracksconf_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(TreeRingComb2_onlyCracks TreeRingComb2_onlyCracks_95 TreeRingComb2_onlyCracks_98 TreeRingComb2_onlyCracks_99 TreeRingComb2_onlyCracks_100)

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights_conf.py --TreeRingConf=${FILE[$SLURM_ARRAY_TASK_ID]} --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5
