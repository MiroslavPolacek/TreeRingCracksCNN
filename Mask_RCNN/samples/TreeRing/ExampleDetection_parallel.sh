#!/usr/bin/env bash

#SBATCH --nodes=2
#SBATCH --array=0-1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --mem=16G
#SBATCH --qos=short
#SBATCH --time=0-08:00:00
#SBATCH --output=ExampleDetections_%a.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

FILE=(treeringcrackscomb2_equalloss20210115T0012/mask_rcnn_treeringcrackscomb2_equalloss_0185.h5 treeringcrackscomb220210114T1540/mask_rcnn_treeringcrackscomb2_0196.h5)

~/.conda/envs/TreeRingCNN/bin/python3 /groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/example_detections.py --weight=/groups/swarts/user/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/${FILE[$SLURM_ARRAY_TASK_ID]}
