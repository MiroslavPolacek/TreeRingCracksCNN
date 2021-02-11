"""
Export images with detected masks
--------------------------
Usage:

THIS TO TEST RUNS
conda activate TreeRingCNN &&
cd ~/Github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing &&
python3 example_detections.py --weight=~/Github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/mask_rcnn_treeringcrackscomb_0284.h5

"""
#######################################################################
#Arguments
#######################################################################
import argparse

    # Parse command line arguments
parser = argparse.ArgumentParser(
        description='Calculate mAP for all epochs')

parser.add_argument('--weight', required=True,
                    metavar="/path/to/weight/folder",
                    help="Path to weight file")

args = parser.parse_args()

#######################################################################
# Prepare packages, models and images
#######################################################################

import os
import sys
import random
import math
import re
import time
import skimage
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize_print
import mrcnn.model as modellib
from mrcnn.model import log

from samples.TreeRing import TreeRingComb2_onlyRing as TreeRing

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = TreeRing.BalloonConfig()
#BALLOON_DIR = args.dataset
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Load validation dataset
#print("Loading dataset")
#dataset = TreeRing.BalloonDataset()
#dataset.load_balloon(BALLOON_DIR, "val")

# Must call before using the dataset
#dataset.prepare()

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
weights_path = args.weight
        # Load weights
print("Loading weights")
model.load_weights(weights_path, by_name=True)
#image_ids = dataset.image_ids

#define class names
class_names = ['BG', 'ring', 'cracks']

#######################################################################
# Print picture example for seprate tricky dataset
#######################################################################
#Image_folder_path = '/home/miroslavp/Pictures/Rings_to_test/transformed_Rins_to-test'
Image_folder_path = '/groups/swarts/user/miroslav.polacek/CNN/transformed_Rins_to-test'


#get folder path and make folder
run_path = args.weight
#print(run_path)
run_split_1 = os.path.split(run_path)
#print(run_split_1)
weight_name = run_split_1[1]
#print('weight_name:', weight_name)
run_ID = os.path.split(run_split_1[0])[1]
#print('run_ID:', run_ID)

model_eval_DIR = os.path.join(ROOT_DIR, 'samples/TreeRing/model_eval_detections')
#model_eval_DIR = '/groups/swarts/user/miroslav.polacek/CNN'
#print(model_eval_DIR)
run_eval_DIR = os.path.join(model_eval_DIR,run_ID)
weight_eval_DIR = os.path.join(run_eval_DIR, weight_name)

if not os.path.exists(run_eval_DIR): #check if it already exists and if not make it
    os.makedirs(run_eval_DIR)

if not os.path.exists(weight_eval_DIR): #check if it already exists and if not make it
    os.makedirs(weight_eval_DIR)


output_path = os.path.join(weight_eval_DIR, 'example_detections')
if not os.path.exists(output_path): #check if it already exists and if not make it
    os.makedirs(output_path)

for image_file in os.listdir(Image_folder_path):
    if image_file.endswith('.tif') or image_file.endswith('.jpg'):
        print(image_file)
        image_path = os.path.join(Image_folder_path, image_file)
        out_image_path = os.path.join(output_path, image_file)
        image = skimage.io.imread(image_path)
        results = model.detect([image], verbose=0)
        r = results[0]
        visualize_print.save_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, out_image_path, r['scores'])
        #plt.figure(figsize=(30,30))
        #plt.axis('off')
        #plt.title(image_file)
        #plt.imshow(to_export)
        #plt.savefig(out_image_path)
        #plt.close()
