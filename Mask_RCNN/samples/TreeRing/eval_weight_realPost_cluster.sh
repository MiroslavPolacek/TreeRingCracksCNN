#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:RTX:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=short
#SBATCH --time=0-01:00:00
#SBATCH --output=eval_real_post.stdout

source activate TreeRingCNN

FILES=(TreeRingComb2_onlyRing_0)

for FILE in ${FILES[@]}
do
  time python3 \
    /users/miroslav.polacek/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights_realPost_simple.py \
    --TreeRingConf=$FILE \
    --dataset=/users/miroslav.polacek/eval_testing/val_new_subset \
    --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5
    --path_out=/users/miroslav.polacek/eval_testing/results
done
