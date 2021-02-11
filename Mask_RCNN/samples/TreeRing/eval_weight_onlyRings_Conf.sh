#!/usr/bin/env bash

#SBATCH --nodes=4
#SBATCH --array=0-3
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --time=0-05:00:00
#SBATCH --output=eval_weight_ringsconf_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(TreeRingComb2_onlyRing_95 TreeRingComb2_onlyRing_98 TreeRingComb2_onlyRing_99 TreeRingComb2_onlyRing_100)

time ~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights_onlyRings_conf.py --TreeRingConf=${FILE[$SLURM_ARRAY_TASK_ID]} --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5