"""
Calculate a lits of mAP at  IoU > 0.5 for every epoch in the folder
--------------------------
Usage:
THIS WAS FOR TESTING ON SIMPLE EXAMPLE
for testing this location
cd /Users/miroslav.polacek/Dropbox\ \(VBC\)/Group\ Folder\ Swarts/Research/CNNRings/Mask_RCNN/samples/TreeRing

run in command line:
cd /Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing &&
conda activate TreeRingCNN &&
python3 eval_epochs_mAP.py --dataset=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/treering  --weight_folder=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcracks20201107T1734

"""
#######################################################################
#Arguments
#######################################################################
import argparse

    # Parse command line arguments
parser = argparse.ArgumentParser(
        description='Calculate mAP for all epochs')

parser.add_argument('--dataset', required=True,
                    metavar="/path/to/ring/dataset/",
                    help='Directory to ring dataset')
parser.add_argument('--weight_folder', required=True,
                    metavar="/path/to/weight/folder",
                    help="Path to weight folder")

args = parser.parse_args()

#######################################################################
#Import necessary stuff
#######################################################################

import os
import sys
import random
import math
import re
import time
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
import mrcnn.model as modellib
from mrcnn.model import log

from samples.TreeRing import TreeRing

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = TreeRing.BalloonConfig()
BALLOON_DIR = args.dataset
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Load validation dataset
dataset = TreeRing.BalloonDataset()
dataset.load_balloon(BALLOON_DIR, "val")

# Must call before using the dataset
dataset.prepare()

# Create model in inference mode

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

#######################################################################
#All necessary functions
#######################################################################

def compute_batch_ap(dataset, image_ids, verbose=1):
    APs = []
    mask_IoU =[]
    for image_id in image_ids:
        # Load image

        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)

        # Run object detection
        #results = model.detect_molded(image[np.newaxis], image_meta[np.newaxis], verbose=0)#gave only one mask
        results = model.detect([image], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        #print(r['scores'])
        #print(r['masks'].shape)
        APs.append(ap)

        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            #print("{:3} {}   AP: {:.2f}".format(
                #meta["image_id"][0], meta["original_image_shape"][0], ap))
    return APs #this outputs mAPs per image. May be usefull to keep it like that
#########################################################################
#Now the looping through weights
#########################################################################
weight_names = []
AP_per_weight = []
AP_per_weight_variance = []

weight_list = os.listdir(args.weight_folder)
short_weight_list = weight_list# I did this just to check for some subset of the weights. E.g. last 100
for f in short_weight_list:
    if f.endswith('h5'):
        #print(f)
        weight_names.append(f)
        weights_path = os.path.join(args.weight_folder,f)
        # Load weights
        print("Loading weights ", f)
        model.load_weights(weights_path, by_name=True)

	# Run validation
        APs = compute_batch_ap(dataset, dataset.image_ids)
        #print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))
        AP_per_weight.append(np.mean(APs)) #change this if you want to have valueas per image and not mean
        AP_per_weight_variance.append(np.var(APs))

#######################################################################
#Save output
#######################################################################
#get folder path and make folder
run_path = args.weight_folder
run_ID = os.path.split(run_path)[1]
model_eval_DIR = os.path.join(ROOT_DIR, 'samples/TreeRing/model_eval')
run_eval_DIR = os.path.join(model_eval_DIR,run_ID)

if not os.path.exists(run_eval_DIR): #check if it already exists and if not make it
    os.makedirs(run_eval_DIR)

#save table
df = pd.DataFrame()
df['weights'] = weight_names
df['mAP'] = AP_per_weight
df['mAP_var'] = AP_per_weight_variance

sorted_df = df.sort_values(by=['mAP'], ascending=False)
sorted_df.to_csv(os.path.join(run_eval_DIR, 'mAP_Weights_table.csv'))
#save graph
plt.plot(range(len(weight_names)), AP_per_weight, label='mAP')
#plt.errorbar(range(len(weight_names)), AP_per_weight, yerr=AP_per_weight_variance, fmt='.k')
plt.ylabel('mAP')
plt.xlabel('Weights')
plt.legend()
#plt.ylim(bottom = 0, top = 1)
#plt.xlim(left = 0, right = 1)
plt.savefig(os.path.join(run_eval_DIR, 'mAP_weights_graph.jpg'))
plt.close()
#print some quick information
print("Evaluation of {} run finished".format(run_ID))
print("Weights", weight_names)
print("APs:", AP_per_weight)
