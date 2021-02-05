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
python3 eval_epochs_mAP.py --dataset=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/treering_mini --weight_folder=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220

run command on my Mint treering_mini
conda activate TreeRingCNN &&
cd /home/miroslavp/Github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing
python3 eval_epochs_mAP_new.py --dataset=/home/miroslavp/Github/TreeRingCracksCNN/Mask_RCNN/datasets/treering_mini --weight_folder=/home/miroslavp/Github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220

run command on my Mint on new_val
conda activate TreeRingCNN &&
cd /home/miroslavp/Github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing
time python3 eval_epochs_mAP_new.py --dataset=/media/miroslavp/MiroAllLife/TreeRingDatasetsBackUp/treering_new --weight_folder=/home/miroslavp/Github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220
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
from csv import writer
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
print("Dataset prepared")
# Create model in inference mode

model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

#######################################################################
#All necessary functions
#######################################################################
def compute_batch_ap(dataset, image_ids, verbose=1):
    APs_general = []
    APs_ring = []
    APs_crack = []
    # loop throgh all val dataset
    for image_id in image_ids:
        # Load image ground truths
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        print(image_id)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP over range 0.5 to 0.95
        r = results[0]
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)

        APs_general.append(ap)
        #print('APs_general', APs_general)
        # get AP values for ring and crack separately
        AP_loop = []
        for i in [1,2]:
            #print("LOOP START", i)
            if gt_mask[:,:,gt_class_id==i].shape[-1] > 0:
                ap = utils.compute_ap_range(gt_bbox[gt_class_id==i],
                gt_class_id[gt_class_id==i], gt_mask[:,:,gt_class_id==i],
                r['rois'][r['class_ids']==i], r['class_ids'][r['class_ids']==i],
                r['scores'][r['class_ids']==i], r['masks'][:,:,r['class_ids']==i], verbose=0)
                AP_loop.append(ap)
                #print(ap)
            else:
                ap = np.nan
                AP_loop.append(ap)
                #print(ap)

        #print('AP_loop', AP_loop)
        APs_ring.append(AP_loop[0])
        APs_crack.append(AP_loop[1])

        if verbose:
            info = dataset.image_info[image_id]
            meta = modellib.parse_image_meta(image_meta[np.newaxis,...])
            #print("{:3} {}   AP: {:.2f}".format(
                #meta["image_id"][0], meta["original_image_shape"][0], ap))

    mAP_general = np.nanmean(APs_general)
    mAP_ring = np.nanmean(APs_ring)
    mAP_crack = np.nanmean(APs_crack)

    return mAP_general, mAP_ring, mAP_crack  #this outputs mAPs per image. May be usefull to keep it like that

# function to append row to existing csv file
def append_row_to_csv(csv_file_path, mAP_to_save):
    with open(csv_file_path, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(mAP_to_save)

#######################################################################
# Output paths
#######################################################################
#get folder path and make folder
run_path = args.weight_folder
run_ID = os.path.basename(run_path)
model_eval_DIR = os.path.join(ROOT_DIR, 'samples/TreeRing/model_eval/')

val_dataset_name = os.path.basename(args.dataset) # to create different folder for different validation set if the same weight run on multiple
run_eval_DIR = os.path.join(model_eval_DIR, run_ID, val_dataset_name)

if not os.path.exists(run_eval_DIR): #check if it already exists and if not make it
    os.makedirs(run_eval_DIR)
# check if csv file already exists or if not create it
csv_file_path = os.path.join(run_eval_DIR, 'mAP_Weights_table.csv')
if not os.path.exists(csv_file_path):
    first_row = ['weight_names', 'mAP_general', 'mAP_ring', 'mAP_crack']
    append_row_to_csv(csv_file_path, first_row)

#########################################################################
#Now the looping through weights
#########################################################################
loaded_csv = pd.read_csv(csv_file_path, index_col=False)
print('loaded_csv',loaded_csv)
finished_weight_names = loaded_csv['weight_names'].tolist()
print('finished_weight_names', finished_weight_names)
weight_list = os.listdir(args.weight_folder)

for f in weight_list:
    if f.endswith('h5') and f not in finished_weight_names:
        weights_path = os.path.join(args.weight_folder,f)
        # Load weights
        print("Loading weights", f)
        model.load_weights(weights_path, by_name=True)

	# Run validation
        mAPs_general, mAPs_ring, mAPs_crack = compute_batch_ap(dataset, dataset.image_ids)
        #print("Mean AP overa {} images: {:.4f}".format(len(APs), np.mean(APs)))

        mAP_to_save = [f, mAPs_general, mAPs_ring, mAPs_crack]
        append_row_to_csv(csv_file_path, mAP_to_save)
#######################################################################
#Sort table by mAP_general and plot the mAP_graph
#######################################################################
# load finished csv and save it sorted
final_csv = pd.read_csv(csv_file_path, index_col=False)
sorted_df = final_csv.sort_values(by=['mAP_general'], ascending=False)
sorted_df.to_csv(os.path.join(run_eval_DIR, 'mAP_Weights_table_sorted.csv'))
weight_names = final_csv['weight_names']
# plot
plt.plot(range(len(weight_names)), final_csv['mAP_general'], label='mAP_general')
plt.plot(range(len(weight_names)), final_csv['mAP_ring'], label='mAP_ring')
plt.plot(range(len(weight_names)), final_csv['mAP_crack'], label='mAP_crack')
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
print("APs:", mAPs_general)
