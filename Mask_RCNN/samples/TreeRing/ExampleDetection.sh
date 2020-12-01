#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --mem=32G
#SBATCH --qos=medium
#SBATCH --time=2-00:00:00
#SBATCH --output=ExampleDetections.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN



~/.conda/envs/TreeRingCNN/bin/python3 /users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/example_detections.py  --weight=/users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackstrans20201117T1416/mask_rcnn_treeringcrackstrans_0038.h5
