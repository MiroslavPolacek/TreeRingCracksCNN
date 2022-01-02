#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=24G
#SBATCH --qos=long
#SBATCH --time=3-00:00:00
#SBATCH --output=eval_real_post_print.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 \
  /users/miroslav.polacek/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights_realPost_print.py \
  --TreeRingConf=TreeRingCombMaskExamples \
  --dataset=/groups/swarts/user/miroslav.polacek/CNN/val_new \
  --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 \
  --path_out=/users/miroslav.polacek/eval_testing/results
