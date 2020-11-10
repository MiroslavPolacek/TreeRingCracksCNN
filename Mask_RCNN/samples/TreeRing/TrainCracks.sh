#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=g
#SBATCH --mem=112G
#SBATCH --qos=medium
#SBATCH --time=2-00:00:00
#SBATCH --output=TrainCracks.stdout

ml load anaconda3/2019.03
source activate ~/.conda/envs/TreeRingCNN

time ~/.conda/envs/TreeRingCNN/bin/python3 /TreeRingCracksCNN/Mask_RCNN/samples/TreeRing/TreeRing.py train --dataset=  --weights=