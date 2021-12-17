#!/bin/bash
source activate TreeRingCNN

FILES=(TreeRingComb2_onlyCracks_0)

for FILE in ${FILES[@]}
do
  time python3 \
    /home/miroslavp/Github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/evaluate_weights_realPostCracks_simple.py \
    --TreeRingConf=$FILE \
    --dataset=/home/miroslavp/Pictures/val_new_subset \
    --weight=/home/miroslavp/Github/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/logs/treeringcrackscomb2_onlycracks20210121T2224/mask_rcnn_treeringcrackscomb2_onlycracks_0522.h5 \
    --path_out=/home/miroslavp/Pictures/new_val_detections/curent_debug_Cracks
done
