"""
Calculate a lits of mAP at IoU > 0.5 and for every epoch in the folder
--------------------------
Usage:

THIS TO TEST RUNS
conda activate TreeRingCNNtest &&
cd /Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing &&
python3 evaluate_weights.py  --dataset=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/treering_mini/  --weight=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/mask_rcnn_treeringcrackscomb_0222.h5

THIS TO TEST RUNS
conda activate TreeRingCNNtest &&
cd /Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing &&
python3 evaluate_weights.py  --dataset=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/datasets/treering/  --weight=/Users/miroslav.polacek/github/TreeRingCracksCNN/Mask_RCNN/logs/treeringcrackscomb20201119T2220/mask_rcnn_treeringcrackscomb_0222.h5

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
parser.add_argument('--weight', required=True,
                    metavar="/path/to/weight/folder",
                    help="Path to weight file")

args = parser.parse_args()

#######################################################################
# Prepare packages, models and images
#######################################################################

import os
import sys
#import re
import skimage
import cv2
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
print("Loading dataset")
dataset = TreeRing.BalloonDataset()
dataset.load_balloon(BALLOON_DIR, "val")

# Must call before using the dataset
dataset.prepare()

# Create model in inference mode
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights
weights_path = args.weight
        # Load weights
print("Loading weights")
model.load_weights(weights_path, by_name=True)
image_ids = dataset.image_ids
# Get all class ids from the dataset

#################################################################################
# Precision and recall for mask, first value of TP...should be for score of 0.5
#################################################################################
def TP_FP_FN_per_score_mask(gt_mask, pred_mask, scores, IoU_treshold):

    #loop scores
    score_range = np.arange(0.5, 1.0, 0.05)

        #print(gt_r)
        #print(pred_r)
    gt_rings = []
    pred_rings = []
    TPs = []
    FPs = []
    FNs = []

    for SR in score_range:
        #print(SR)
        score_ids = np.where(scores > SR)[0] #Ids for predictions above certain score threshold
        #print(score_ids)
        mask_SR = np.take(pred_mask, score_ids, axis=2)
        #print('mask_SR.shape:', mask_SR.shape)

        mask_matrix = utils.compute_overlaps_masks(gt_mask, mask_SR)
        #print("mask_matrix", mask_matrix)
        #for every score range callculate TP, ...append by the socre ranges
        # making binary numpy array with IoU treshold
        mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
        #print (mask_matrix_binary)

        #GT rings and predicted rigs
        #print(mask_matrix.shape)
        if mask_matrix.shape[0]==0:
            TPs.append(0)
            FPs.append(0)
            FNs.append(0)
        else:
            gt_r = len(mask_matrix)
            pred_r = len(mask_matrix[0])

            #TP
            sum_truth = np.sum(mask_matrix_binary, axis=1)
            sum_truth_binary = np.where(sum_truth > 0, 1, 0)
            TP = np.sum(sum_truth_binary)
            TPs.append(TP)
            #print('TPs:', TPs)
            #FP
            sum_pred = np.sum(mask_matrix_binary, axis=0)
            sum_pred_binary = np.where(sum_pred > 0, 1, 0)
            FP = pred_r - np.sum(sum_pred_binary)
            FPs.append(FP)
            #print('FP:', FP)
            #FN
            FN = gt_r - TP
            FNs.append(FN)
        #print('FN:', FN)
    #put together and sum up TP...per range

    return TPs, FPs, FNs, score_range

#######################################################################
# mAP graph
#######################################################################
def compute_ap_range_list(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    APlist = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls, overlaps =\
            utils.compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask, iou_threshold=iou_threshold)
        APlist.append(ap)

        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))

    #print(APlist)
    return APlist
##########################################################################
# Turn flat combined mask into array with layer per every mask
##########################################################################
def modify_flat_mask(mask):
    #### identify polygons with opencv
    binary_mask = np.where(mask >= 1, 255, 0) # this part can be cleaned to remove some missdetections setting condition for >2
    #plt.imshow(binary_mask)
    #plt.show()

    uint8binary = binary_mask.astype(np.uint8).copy()

    #gray_image = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    # Older version of openCV has slightly different syntax i adjusted for it here
    if int(cv2.__version__.split(".")[0]) < 4:
        _, contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    else:
        contours, _ = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #print('contour_shape:', len(contours))
    #print('contours.shape:', contours)
    #### in a loop through polygons turn every one into binary mask of propper dimension and append
    imgheight, imgwidth = mask.shape[:2]


    clean_contours = []
    for i in range(len(contours)):
        #print('contours[i]:',contours[i])
        #remove too small contours because they do not make sence
        rect = cv2.minAreaRect(contours[i])
        dim1, dim2 = rect[1]
        dim_max = max([dim1, dim2])
        if dim_max > imgheight/3:
            clean_contours.append(contours[i])

    #print('len_clean_contours', len(clean_contours))

    # create empty mask
    result_mask = np.zeros([imgheight, imgwidth, len(clean_contours)], dtype=np.uint8)

    for i in range(len(clean_contours)):
        #print('i', i)
        #print('len clean contours[i]', len(clean_contours[i]))
        x_points = []
        y_points = []
        for j in range(len(clean_contours[i])):
            #print('contij:', clean_contours[i][j])
            [[xc, yc]] = clean_contours[i][j]
            x_points.append(xc)
            y_points.append(yc)
        #print('len x', len(x_points))
        # Get indexes of pixels inside the polygon and set them to 1
        #print('x:', x_points)
        #print('y:', y_points)
        rr, cc = skimage.draw.polygon(y_points, x_points)
        #print('rr', rr)
        result_mask[rr, cc, i] = 1
        #print('res_mask', result_mask[:,:,0])


    return result_mask

###########################################################################
# Calculate AP gorup of indexes. General and per class.
###########################################################################
def mAP_group(image, gt_class_id, gt_bbox, gt_mask, pred_bbox, pred_mask, pred_class_id, pred_scores):
    AP_general = []
    AP_names = ["mAP", "AP50", "APlist","mAP_ring", "AP50_ring", "APlist_ring","mAP_crack", "AP50_crack", "APlist_crack","mAP_resin", "AP50_resin", "APlist_resin","mAP_pith", "AP50_pith", "APlist_pith" ]
    # if no mask is detected
    #if pred_mask.shape[-1] == 0:
        #AP_general = [0,0,[0]*10]*5
        #print("mAP_group gave zeroes for this image")
    #else:

    # mAP, AP50 for all classes
    AP_list = compute_ap_range_list(gt_bbox, gt_class_id, gt_mask, pred_bbox, pred_class_id, pred_scores, pred_mask, verbose=0)
    mAP = np.array(AP_list).mean()
    #print("mAP for this image", mAP)
    AP50 = AP_list[0]
    AP_general = [mAP, AP50, AP_list]
    # for each class_id
    for i in range(1,5):
        #print("LOOP START", i)
        if gt_mask[:,:,gt_class_id==i].shape[-1] > 0:
            AP_list = compute_ap_range_list(gt_bbox[gt_class_id==i], gt_class_id[gt_class_id==i], gt_mask[:,:,gt_class_id==i], pred_bbox[pred_class_id==i], pred_class_id[pred_class_id==i], pred_scores[pred_class_id==i], pred_mask[:,:,pred_class_id==i], verbose=0)
            mAP = np.array(AP_list).mean()
            AP50 = AP_list[0]
            #print("AP50 fopr category {} is {}".format(i, AP50))
        else:
            mAP = np.nan
            AP50 = np.nan
            AP_list = [np.nan]*10

        #print("mAPlist category {} is: {}".format(i, AP_list))
        #print("gt_masks", gt_mask[:,:,gt_class_id==i].shape[-1])
        #print("predicted masks", pred_mask[:,:,pred_class_id==i].shape[-1])
        #print("AP50 out of if else for category {} is {}".format(i, AP50))
        AP_general.extend([mAP, AP50, AP_list])

    return AP_general, AP_names #mAP_group_values #should be a list of lists of names and values
###########################################################################
# Calculate TP_FP_NF_per_score_mask. General and per class.
###########################################################################
def TP_FP_FN_group(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5):
    TP_FP_FN_general = []
    TP_FP_FN_names = ["score_range", "TP", "FP", "FN","TP_ring", "FP_ring", "FN_ring","TP_crack", "FP_crack", "FN_crack","TP_resin", "FP_resin", "FN_resin", "TP_pith", "FP_pith", "FN_pith"]
    # if no mask is detected
    if pred_mask.shape[-1] == 0:
         TP_FP_FN_general = [[0]*10]*16
    else:
        # for all classes
        TP, FP, FN, score_range = TP_FP_FN_per_score_mask(gt_mask, pred_mask, pred_scores, IoU_treshold=IoU_treshold)
        TP_FP_FN_general = [score_range, TP, FP, FN]
        for i in range(1,5):
            TP, FP, FN, score_range = TP_FP_FN_per_score_mask(gt_mask[:,:,gt_class_id==i], pred_mask[:,:,pred_class_id==i], pred_scores[pred_class_id==i], IoU_treshold=IoU_treshold)
            TP_FP_FN_general.extend([TP, FP, FN])

    return TP_FP_FN_general, TP_FP_FN_names

###########################################################################
# Calculate IoU general and per class.
###########################################################################
def IoU_group(gt_mask, gt_class_id, pred_mask, pred_class_id):
    IoU_general = []
    IoU_names = ["IoU", "IoU_ring", "IoU_crack", "IoU_resin", "IoU_pith"]
    # if no mask is detected
    if pred_mask.shape[-1] == 0:
        IoU_general = [0]*5
    else:
        IoU= utils.compute_overlaps_masks(gt_mask, pred_mask)
        IoU = np.nan_to_num(np.mean(IoU)) #change nans to 0
        IoU_general = [IoU]
        for i in range(1,5):
            IoU= utils.compute_overlaps_masks(gt_mask[:,:,gt_class_id==i], pred_mask[:,:,pred_class_id==i])
            IoU = np.nan_to_num(np.mean(IoU)) #change nans to 0
            IoU_general.append(IoU)

    return IoU_general, IoU_names
###########################################################################
# now calculate values for whole dataset
###########################################################################
# variables to calculate
## AP group
mAP = []
AP50 = []
APlist = []
mAP_ring = []
AP50_ring = []
APlist_ring = []
mAP_crack = []
AP50_crack = []
APlist_crack = []
mAP_resin = []
AP50_resin = []
APlist_resin =[]
mAP_pith = []
AP50_pith = []
APlist_pith = []

## TP_FP_FN_group
TP = []
FP = []
FN = []
TP_ring = []
FP_ring = []
FN_ring = []
TP_crack = []
FP_crack = []
FN_crack = []
TP_resin = []
FP_resin = []
FN_resin = []
TP_pith = []
FP_pith = []
FN_pith = []

## IoU_group
IoU = []
IoU_ring = []
IoU_crack = []
IoU_resin = []
IoU_pith = []

#90 DEGREE ROTATION

## AP group _90
mAP_90 = []
AP50_90 = []
APlist_90 = []
mAP_ring_90 = []
AP50_ring_90 = []
APlist_ring_90 = []
mAP_crack_90 = []
AP50_crack_90 = []
APlist_crack_90 = []
mAP_resin_90 = []
AP50_resin_90 = []
APlist_resin_90 =[]
mAP_pith_90 = []
AP50_pith_90 = []
APlist_pith_90 = []

## TP_FP_FN_group
TP_90 = []
FP_90 = []
FN_90 = []
TP_ring_90 = []
FP_ring_90 = []
FN_ring_90 = []
TP_crack_90 = []
FP_crack_90 = []
FN_crack_90 = []
TP_resin_90 = []
FP_resin_90 = []
FN_resin_90 = []
TP_pith_90 = []
FP_pith_90 = []
FN_pith_90 = []

## IoU_group
IoU_90 = []
IoU_ring_90 = []
IoU_crack_90 = []
IoU_resin_90 = []
IoU_pith_90 = []

#45 DEGREE ROTATION
mAP_45 = []
AP50_45 = []
APlist_45 = []
mAP_ring_45 = []
AP50_ring_45 = []
APlist_ring_45 = []
mAP_crack_45 = []
AP50_crack_45 = []
APlist_crack_45 = []
mAP_resin_45 = []
AP50_resin_45 = []
APlist_resin_45 =[]
mAP_pith_45 = []
AP50_pith_45 = []
APlist_pith_45 = []

## TP_FP_FN_group
TP_45 = []
FP_45 = []
FN_45 = []
TP_ring_45 = []
FP_ring_45 = []
FN_ring_45= []
TP_crack_45 = []
FP_crack_45 = []
FN_crack_45 = []
TP_resin_45 = []
FP_resin_45 = []
FN_resin_45 = []
TP_pith_45 = []
FP_pith_45 = []
FN_pith_45 = []

## IoU_group
IoU_45 = []
IoU_ring_45 = []
IoU_crack_45 = []
IoU_resin_45 = []
IoU_pith_45 = []

#COMBINED MASK
IoU_combined_mask = []
TPs_combined = []
FPs_combined = []
FNs_combined = []
# Main structure
for image_id in image_ids:

    ## Load the ground truth for the image
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    print('EVALUATING IMAGE:', image_id)
    #print('image shape:', image.shape)
    imgheight = image.shape[0]

###### Detect image in normal orientation
    results = model.detect([image], verbose=0)
    r = results[0]
    mask_normal = r['masks'] # for the combined mask at the end
    mask_normal_classes = r['class_ids']
    #print("check shapes", gt_class_id, gt_mask[:,:,gt_class_id==1].shape)
    # pass this r to the functions
    #(image, image_meta, gt_class_id, gt_bbox, gt_mask, pred_bbox, pred_mask, pred_class_id, pred_scores)
    AP_general, AP_names = mAP_group(image, gt_class_id, gt_bbox, gt_mask, r['rois'], r['masks'], r['class_ids'], r['scores'])

    mAP.append(AP_general[AP_names.index("mAP")])
    AP50.append(AP_general[AP_names.index("AP50")])
    APlist.append(AP_general[AP_names.index("APlist")])
    mAP_ring.append(AP_general[AP_names.index("mAP_ring")])
    AP50_ring.append(AP_general[AP_names.index("AP50_ring")])
    APlist_ring.append(AP_general[AP_names.index("APlist_ring")])
    mAP_crack.append(AP_general[AP_names.index("mAP_crack")])
    AP50_crack.append(AP_general[AP_names.index("AP50_crack")])
    APlist_crack.append(AP_general[AP_names.index("APlist_crack")])
    #print("APlist_crack", APlist_crack)
    mAP_resin.append(AP_general[AP_names.index("mAP_resin")])
    AP50_resin.append(AP_general[AP_names.index("AP50_resin")])
    APlist_resin.append(AP_general[AP_names.index("APlist_resin")])
    #print("APlist_resin", APlist_resin)
    mAP_pith.append(AP_general[AP_names.index("mAP_pith")])
    AP50_pith.append(AP_general[AP_names.index("AP50_pith")])
    APlist_pith.append(AP_general[AP_names.index("APlist_pith")])
    #print("APlist_pith", APlist_pith)
    #print("mAP", mAP)
    #print("AP50", AP50)
    #print("mAP_ring", mAP_ring)
    #print("AP50_ring", AP50_ring)

    #(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5)
    TP_general, TP_names = TP_FP_FN_group(gt_mask, gt_class_id, r['masks'], r['class_ids'], r['scores'], IoU_treshold=0.5)

    TP.append(TP_general[TP_names.index("TP")])
    FP.append(TP_general[TP_names.index("FP")])
    FN.append(TP_general[TP_names.index("FN")])
    TP_ring.append(TP_general[TP_names.index("TP_ring")])
    FP_ring.append(TP_general[TP_names.index("FP_ring")])
    FN_ring.append(TP_general[TP_names.index("FN_ring")])
    TP_crack.append(TP_general[TP_names.index("TP_crack")])
    FP_crack.append(TP_general[TP_names.index("FP_crack")])
    FN_crack.append(TP_general[TP_names.index("FN_crack")])
    TP_resin.append(TP_general[TP_names.index("TP_resin")])
    FP_resin.append(TP_general[TP_names.index("FP_resin")])
    FN_resin.append(TP_general[TP_names.index("FN_resin")])
    TP_pith.append(TP_general[TP_names.index("TP_pith")])
    FP_pith.append(TP_general[TP_names.index("FP_pith")])
    FN_pith.append(TP_general[TP_names.index("FN_pith")])
    #print("TP_ring", TP_ring)
    #print("FP_crack", FP_crack)
    #print("FP_ring", FP_ring)

    IoU_general, IoU_names = IoU_group(gt_mask, gt_class_id, r['masks'], r['class_ids'])

    IoU.append(IoU_general[IoU_names.index("IoU")])
    IoU_ring.append(IoU_general[IoU_names.index("IoU_ring")])
    IoU_crack.append(IoU_general[IoU_names.index("IoU_crack")])
    IoU_resin.append(IoU_general[IoU_names.index("IoU_resin")])
    IoU_pith.append(IoU_general[IoU_names.index("IoU_pith")])
    #print("IoU", IoU)
    #print("IoU_crack", IoU_crack)
    #print("IoU_resin", IoU_resin)

###### DETECT IMAGE 90degree

    ### rotate image, detect, rotate mask back
    image_90 = skimage.transform.rotate(image, 90, preserve_range=True).astype(np.uint8)
    #print('image90 shape:', image_90.shape)
    #plt.imshow(image_90)
    #plt.show()
    results = model.detect([image_90], verbose=0)
    r = results[0]

    # rotate mask back
    mask_90_back = np.rot90(r['masks'], k=-1)
    mask_90_classes = r['class_ids']
    # get the associated bbox
    extracted_bboxes = utils.extract_bboxes(mask_90_back)

    ###calculate all the stuff
    AP_general, AP_names = mAP_group(image, gt_class_id, gt_bbox, gt_mask, extracted_bboxes, mask_90_back, r['class_ids'], r['scores'])

    mAP_90.append(AP_general[AP_names.index("mAP")])
    AP50_90.append(AP_general[AP_names.index("AP50")])
    APlist_90.append(AP_general[AP_names.index("APlist")])
    mAP_ring_90.append(AP_general[AP_names.index("mAP_ring")])
    AP50_ring_90.append(AP_general[AP_names.index("AP50_ring")])
    APlist_ring_90.append(AP_general[AP_names.index("APlist_ring")])
    mAP_crack_90.append(AP_general[AP_names.index("mAP_crack")])
    AP50_crack_90.append(AP_general[AP_names.index("AP50_crack")])
    APlist_crack_90.append(AP_general[AP_names.index("APlist_crack")])
    mAP_resin_90.append(AP_general[AP_names.index("mAP_resin")])
    AP50_resin_90.append(AP_general[AP_names.index("AP50_resin")])
    APlist_resin_90.append(AP_general[AP_names.index("APlist_resin")])
    mAP_pith_90.append(AP_general[AP_names.index("mAP_pith")])
    AP50_pith_90.append(AP_general[AP_names.index("AP50_pith")])
    APlist_pith_90.append(AP_general[AP_names.index("APlist_pith")])
    #print("mAP_90", mAP_90)
    #print("AP50_90", AP50_90)
    #print("mAP_ring_90", mAP_ring_90)
    #print("AP50_ring_90", AP50_ring_90)

    #(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5)
    TP_general, TP_names = TP_FP_FN_group(gt_mask, gt_class_id, mask_90_back, r['class_ids'], r['scores'], IoU_treshold=0.5)

    TP_90.append(TP_general[TP_names.index("TP")])
    FP_90.append(TP_general[TP_names.index("FP")])
    FN_90.append(TP_general[TP_names.index("FN")])
    TP_ring_90.append(TP_general[TP_names.index("TP_ring")])
    FP_ring_90.append(TP_general[TP_names.index("FP_ring")])
    FN_ring_90.append(TP_general[TP_names.index("FN_ring")])
    TP_crack_90.append(TP_general[TP_names.index("TP_crack")])
    FP_crack_90.append(TP_general[TP_names.index("FP_crack")])
    FN_crack_90.append(TP_general[TP_names.index("FN_crack")])
    TP_resin_90.append(TP_general[TP_names.index("TP_resin")])
    FP_resin_90.append(TP_general[TP_names.index("FP_resin")])
    FN_resin_90.append(TP_general[TP_names.index("FN_resin")])
    TP_pith_90.append(TP_general[TP_names.index("TP_pith")])
    FP_pith_90.append(TP_general[TP_names.index("FP_pith")])
    FN_pith_90.append(TP_general[TP_names.index("FN_pith")])
    #print("TP_ring_90", TP_ring_90)
    #print("FP_crack_90", FP_crack_90)
    #print("FP_ring_90", FP_ring_90)

    IoU_general, IoU_names = IoU_group(gt_mask, gt_class_id, mask_90_back, r['class_ids'])

    IoU_90.append(IoU_general[IoU_names.index("IoU")])
    IoU_ring_90.append(IoU_general[IoU_names.index("IoU_ring")])
    IoU_crack_90.append(IoU_general[IoU_names.index("IoU_crack")])
    IoU_resin_90.append(IoU_general[IoU_names.index("IoU_resin")])
    IoU_pith_90.append(IoU_general[IoU_names.index("IoU_pith")])
    #print("IoU_90", IoU_90)
    #print("IoU_crack_90", IoU_crack_90)
    #print("IoU_resin_90", IoU_resin_90)


###### DETECT IMAGE 45degree

    ### rotate image, detect
    image_45 = skimage.transform.rotate(image, angle = 45, resize=True, preserve_range=True).astype(np.uint8)
    results = model.detect([image_45], verbose=0)
    r = results[0]
    if r['masks'].shape[-1] == 0:
        mask_45_back = np.zeros(shape=(imgheight, imgheight,0))
        mask_45_classes = [0]
    else:
        # rotate the mask back
        maskr2_back = skimage.transform.rotate(r['masks'], angle = -45, resize=False)
        #crop to the right size
        imgheight, imgwidth = image.shape[:2]
        imgheight2, imgwidth2 = maskr2_back.shape[:2]
        #print('img_shape:', image.shape)
        #print('img_45_shape:', maskr2_back.shape)
        to_crop = int((imgheight2 - imgheight)/2)
        mask_45_back = maskr2_back[to_crop: (to_crop+int(imgheight)), to_crop: (to_crop+int(imgheight))]
        mask_45_classes = r['class_ids']
    # extract bounding boxes based from the masks
    extracted_bboxes = utils.extract_bboxes(mask_45_back)

    ###calculate all the stuff
    AP_general, AP_names = mAP_group(image, gt_class_id, gt_bbox, gt_mask, extracted_bboxes, mask_45_back, r['class_ids'], r['scores'])

    mAP_45.append(AP_general[AP_names.index("mAP")])
    AP50_45.append(AP_general[AP_names.index("AP50")])
    APlist_45.append(AP_general[AP_names.index("APlist")])
    mAP_ring_45.append(AP_general[AP_names.index("mAP_ring")])
    AP50_ring_45.append(AP_general[AP_names.index("AP50_ring")])
    APlist_ring_45.append(AP_general[AP_names.index("APlist_ring")])
    mAP_crack_45.append(AP_general[AP_names.index("mAP_crack")])
    AP50_crack_45.append(AP_general[AP_names.index("AP50_crack")])
    APlist_crack_45.append(AP_general[AP_names.index("APlist_crack")])
    mAP_resin_45.append(AP_general[AP_names.index("mAP_resin")])
    AP50_resin_45.append(AP_general[AP_names.index("AP50_resin")])
    APlist_resin_45.append(AP_general[AP_names.index("APlist_resin")])
    mAP_pith_45.append(AP_general[AP_names.index("mAP_pith")])
    AP50_pith_45.append(AP_general[AP_names.index("AP50_pith")])
    APlist_pith_45.append(AP_general[AP_names.index("APlist_pith")])
    #print("mAP_45", mAP_45)
    #print("AP50_45", AP50_45)
    #print("mAP_ring_45", mAP_ring_45)
    #print("AP50_ring_45", AP50_ring_45)

    #(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_treshold=0.5)
    TP_general, TP_names = TP_FP_FN_group(gt_mask, gt_class_id, mask_45_back, r['class_ids'], r['scores'], IoU_treshold=0.5)

    TP_45.append(TP_general[TP_names.index("TP")])
    FP_45.append(TP_general[TP_names.index("FP")])
    FN_45.append(TP_general[TP_names.index("FN")])
    TP_ring_45.append(TP_general[TP_names.index("TP_ring")])
    FP_ring_45.append(TP_general[TP_names.index("FP_ring")])
    FN_ring_45.append(TP_general[TP_names.index("FN_ring")])
    TP_crack_45.append(TP_general[TP_names.index("TP_crack")])
    FP_crack_45.append(TP_general[TP_names.index("FP_crack")])
    FN_crack_45.append(TP_general[TP_names.index("FN_crack")])
    TP_resin_45.append(TP_general[TP_names.index("TP_resin")])
    FP_resin_45.append(TP_general[TP_names.index("FP_resin")])
    FN_resin_45.append(TP_general[TP_names.index("FN_resin")])
    TP_pith_45.append(TP_general[TP_names.index("TP_pith")])
    FP_pith_45.append(TP_general[TP_names.index("FP_pith")])
    FN_pith_45.append(TP_general[TP_names.index("FN_pith")])
    #print("TP_ring_45", TP_ring_45)
    #print("FP_crack_45", FP_crack_45)
    #print("FP_ring_45", FP_ring_45)

    IoU_general, IoU_names = IoU_group(gt_mask, gt_class_id, mask_45_back, r['class_ids'])

    IoU_45.append(IoU_general[IoU_names.index("IoU")])
    IoU_ring_45.append(IoU_general[IoU_names.index("IoU_ring")])
    IoU_crack_45.append(IoU_general[IoU_names.index("IoU_crack")])
    IoU_resin_45.append(IoU_general[IoU_names.index("IoU_resin")])
    IoU_pith_45.append(IoU_general[IoU_names.index("IoU_pith")])
    #print("IoU_45", IoU_45)
    #print("IoU_crack_45", IoU_crack_45)
    #print("IoU_resin_45", IoU_resin_45)



###### COMBINE ALL THE MASKS of class RING TO ONE AND CLACULATE IoU for Ring
    #normal flatten
    mask_normal_flat = np.zeros(shape=(imgheight, imgheight))
    mask_normal = mask_normal[:,:, mask_normal_classes==1] # to get only masks for rings
    nmasks = mask_normal.shape[2]
    if nmasks == 0:
        mask_normal_flat = np.zeros(shape=(imgheight, imgheight))
    else:
        for m in range(0,nmasks):
            mask_normal_flat = mask_normal_flat + mask_normal[:,:,m]
    #print("nmasks", nmasks)
    #print("mask_normal_flat", mask_normal_flat.shape)
    #plt.imshow(mask_normal_flat)
    #plt.show()
    #90d flatten
    mask_90_flat = np.zeros(shape=(imgheight, imgheight))
    mask_90_back = mask_90_back[:,:,mask_90_classes==1] # to get only masks for rings
    nmasks = mask_90_back.shape[2]
    if nmasks == 0:
        mask_90_flat = np.zeros(shape=(imgheight, imgheight))
    else:
        for m in range(0,nmasks):
            mask_90_flat = mask_90_flat + mask_90_back[:,:,m]
    #print("nmasks", nmasks)
    #print("mask_90_flat", mask_90_flat.shape)
    #45d flatten
    mask_45_flat = np.zeros(shape=(imgheight, imgheight))
    mask_45_back = mask_45_back[:,:,mask_45_classes==1] # to get only masks for rings
    nmasks = mask_45_back.shape[2]
    if nmasks == 0:
        mask_45_flat = np.zeros(shape=(imgheight, imgheight))
    else:
        for m in range(0,nmasks):
            mask_45_flat = mask_45_flat + mask_45_back[:,:,m]
    #print("nmasks", nmasks)
    #print("mask_45_flat", mask_45_flat.shape)
    #combine to one
    combined_mask = mask_normal_flat + mask_90_flat + mask_45_flat
    #print("combined_mask", combined_mask.shape)
    #plt.imshow(combined_mask)
    #plt.show()

    #flatten ground truth mask
    gt_mask_flat = np.zeros(shape=(imgheight, imgheight))
    gt_mask = gt_mask[:,:,gt_class_id==1]
    nmasks = gt_mask.shape[2]
    for m in range(0,nmasks):
        gt_mask_flat = gt_mask_flat + gt_mask[:,:,m]
    #calcumate IoU
    combined_mask_binary = np.where(combined_mask > 0, 1, 0)
    #print("combined_mask_binary",combined_mask_binary.shape)
    combined_mask_binary = np.reshape(combined_mask_binary, (1024,1024,1))

    #print('combined_mask_shape:', combined_mask_binary.shape)
    gt_mask_flat_binary = np.where(gt_mask_flat > 0, 1, 0)
    #print(gt_mask_flat_binary.shape)
    gt_mask_flat_binary = np.reshape(gt_mask_flat_binary, (1024,1024,1))
    print('gt_mask_shape:', gt_mask_flat_binary.shape)
    IoU_combined_mask.append(utils.compute_overlaps_masks(gt_mask_flat_binary, combined_mask_binary))

    #IoU_combined_mask = np.mean(IoU_combined_mask)
    #print('IoU_combined:', IoU_combined_mask)

####### Try to separate combined masks into layers
    separated_mask = modify_flat_mask(combined_mask)
    #print('separated_mask_shape', separated_mask.shape)

    #print(IoU_m)
    #plt.imshow(gt_mask[:,:,3])
    #plt.show()
    #plt.imshow(separated_mask[:,:,0])
    #plt.show()
    mask_matrix = utils.compute_overlaps_masks(gt_mask, separated_mask)
    #print("mask_matrix.shape", mask_matrix.shape)
    # making binary numpy array with IoU treshold
    IoU_treshold = 0.5 # i set it less because combined mask is bigger and it does not matter i think
    mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
    #print (mask_matrix_binary)


    #GT rings and predicted rigs
    gt_r = len(mask_matrix)
    pred_r = len(mask_matrix[0])

    #TP
    sum_truth = np.sum(mask_matrix_binary, axis=1)
    sum_truth_binary = np.where(sum_truth > 0, 1, 0)
    TP_comb = np.sum(sum_truth_binary)
    TPs_combined.append(TP_comb)
    #print('TP:', TP)
    #FP
    sum_pred = np.sum(mask_matrix_binary, axis=0)
    sum_pred_binary = np.where(sum_pred > 0, 1, 0)
    FP_comb = pred_r - np.sum(sum_pred_binary)
    FPs_combined.append(FP_comb)
    #print('FP:', FP)
    #FN
    FN_comb = gt_r - TP_comb
    FNs_combined.append(FN_comb)
#print("IoU_combined_mask",IoU_combined_mask)

#calculate averages for all images
#0
## AP group
mAP = np.nanmean(mAP)
print("mAP", mAP)
AP50 = np.nanmean(AP50)
APlist = np.nanmean(APlist, axis=0)
mAP_ring = np.nanmean(mAP_ring)
AP50_ring = np.nanmean(AP50_ring)
APlist_ring = np.nanmean(APlist_ring, axis=0)
print("mAP_crack", mAP_crack)
print("mAP_crack_45", mAP_crack_45)
print("mAP_crack_90", mAP_crack_90)
mAP_crack = np.nanmean(mAP_crack)
print("mAP_crack", mAP_crack)
print("AP50_crack", AP50_crack)
print("AP50_crack_45", AP50_crack_45)
print("AP50_crack_90", AP50_crack_90)
AP50_crack = np.nanmean(AP50_crack)
APlist_crack = np.nanmean(APlist_crack, axis=0)
mAP_resin = np.nanmean(mAP_resin)
AP50_resin = np.nanmean(AP50_resin)
APlist_resin = np.nanmean(APlist_resin, axis=0)
mAP_pith = np.nanmean(mAP_pith)
AP50_pith = np.nanmean(AP50_pith)
APlist_pith = np.nanmean(APlist_pith, axis=0)

## TP_FP_FN_group
TP = np.array(np.sum(TP, axis=0))
FP = np.array(np.sum(FP, axis=0))
FN = np.array(np.sum(FN, axis=0))
### calculate sensitivity and precission
SEN = TP/(TP+FN)
PREC = TP/(TP+FP)
#print("SEN", SEN)
#print("PREC", PREC)

TP_ring = np.array(np.sum(TP_ring, axis=0))
FP_ring = np.array(np.sum(FP_ring, axis=0))
FN_ring = np.array(np.sum(FN_ring, axis=0))
### calculate sensitivity and precission
SEN_ring = TP_ring/(TP_ring+FN_ring)
PREC_ring = TP_ring/(TP_ring+FP_ring)

TP_crack = np.array(np.sum(TP_crack, axis=0))
FP_crack = np.array(np.sum(FP_crack, axis=0))
FN_crack = np.array(np.sum(FN_crack, axis=0))
### calculate sensitivity and precission
SEN_crack = TP_crack/(TP_crack+FN_crack)
PREC_crack = TP_crack/(TP_crack+FP_crack)

TP_resin = np.array(np.sum(TP_resin, axis=0))
FP_resin = np.array(np.sum(FP_resin, axis=0))
FN_resin = np.array(np.sum(FN_resin, axis=0))
### calculate sensitivity and precission
SEN_resin = TP_resin/(TP_resin+FN_resin)
PREC_resin = TP_resin/(TP_resin+FP_resin)
#print("SEN_resin", SEN_resin)
TP_pith = np.array(np.sum(TP_pith, axis=0))
FP_pith = np.array(np.sum(FP_pith, axis=0))
FN_pith = np.array(np.sum(FN_pith, axis=0))
### calculate sensitivity and precission
SEN_pith = TP_pith/(TP_pith+FN_pith)
PREC_pith = TP_pith/(TP_pith+FP_pith)

## IoU_group
IoU = np.mean(IoU)
IoU_ring = np.mean(IoU_ring)
IoU_crack = np.mean(IoU_crack)
IoU_resin = np.mean(IoU_resin)
IoU_pith = np.mean(IoU_pith)

# 90
## AP group
mAP_90 = np.nanmean(mAP_90)
#print("mAP_90", mAP_90)
AP50_90 = np.nanmean(AP50_90)
APlist_90 = np.nanmean(APlist_90, axis=0)
mAP_ring_90 = np.nanmean(mAP_ring_90)
AP50_ring_90 = np.nanmean(AP50_ring_90)
APlist_ring_90 = np.nanmean(APlist_ring_90, axis=0)
mAP_crack_90 = np.nanmean(mAP_crack_90)
AP50_crack_90 = np.nanmean(AP50_crack_90)
APlist_crack_90 = np.nanmean(APlist_crack_90, axis=0)
mAP_resin_90 = np.nanmean(mAP_resin_90)
AP50_resin_90 = np.nanmean(AP50_resin_90)
APlist_resin_90 = np.nanmean(APlist_resin_90, axis=0)
mAP_pith_90 = np.nanmean(mAP_pith_90)
AP50_pith_90 = np.nanmean(AP50_pith_90)
APlist_pith_90 = np.nanmean(APlist_pith_90, axis=0)

## TP_FP_FN_group
TP_90 = np.array(np.sum(TP_90, axis=0))
FP_90 = np.array(np.sum(FP_90, axis=0))
FN_90 = np.array(np.sum(FN_90, axis=0))
### calculate sensitivity and precission
SEN_90 = TP_90/(TP_90+FN_90)
PREC_90 = TP_90/(TP_90+FP_90)
#print("SEN_90", SEN_90)
#print("PREC_90", PREC_90)

TP_ring_90 = np.array(np.sum(TP_ring_90, axis=0))
FP_ring_90 = np.array(np.sum(FP_ring_90, axis=0))
FN_ring_90 = np.array(np.sum(FN_ring_90, axis=0))
### calculate sensitivity and precission
SEN_ring_90 = TP_ring_90/(TP_ring_90+FN_ring_90)
PREC_ring_90 = TP_ring_90/(TP_ring_90+FP_ring_90)

TP_crack_90 = np.array(np.sum(TP_crack_90, axis=0))
FP_crack_90 = np.array(np.sum(FP_crack_90, axis=0))
FN_crack_90 = np.array(np.sum(FN_crack_90, axis=0))
### calculate sensitivity and precission
SEN_crack_90 = TP_crack_90/(TP_crack_90+FN_crack_90)
PREC_crack_90 = TP_crack_90/(TP_crack_90+FP_crack_90)

TP_resin_90 = np.array(np.sum(TP_resin_90, axis=0))
FP_resin_90 = np.array(np.sum(FP_resin_90, axis=0))
FN_resin_90 = np.array(np.sum(FN_resin_90, axis=0))
### calculate sensitivity and precission
SEN_resin_90 = TP_resin_90/(TP_resin_90+FN_resin_90)
PREC_resin_90 = TP_resin_90/(TP_resin_90+FP_resin_90)

TP_pith_90 = np.array(np.sum(TP_pith_90, axis=0))
FP_pith_90 = np.array(np.sum(FP_pith_90, axis=0))
FN_pith_90 = np.array(np.sum(FN_pith_90, axis=0))
### calculate sensitivity and precission
SEN_pith_90 = TP_pith_90/(TP_pith_90+FN_pith_90)
PREC_pith_90 = TP_pith_90/(TP_pith_90+FP_pith_90)

## IoU_group
IoU_90 = np.mean(IoU_90)
IoU_ring_90 = np.mean(IoU_ring_90)
IoU_crack_90 = np.mean(IoU_crack_90)
IoU_resin_90 = np.mean(IoU_resin_90)
IoU_pith_90 = np.mean(IoU_pith_90)

# 45
mAP_45 = np.nanmean(mAP_45)
#print("mAP_45", mAP_45)
AP50_45 = np.nanmean(AP50_45)
APlist_45 = np.nanmean(APlist_45, axis=0)
mAP_ring_45 = np.nanmean(mAP_ring_45)
AP50_ring_45 = np.nanmean(AP50_ring_45)
APlist_ring_45 = np.nanmean(APlist_ring_45, axis=0)
mAP_crack_45 = np.nanmean(mAP_crack_45)
AP50_crack_45 = np.nanmean(AP50_crack_45)
APlist_crack_45 = np.nanmean(APlist_crack_45, axis=0)
mAP_resin_45 = np.nanmean(mAP_resin_45)
AP50_resin_45 = np.nanmean(AP50_resin_45)
APlist_resin_45 = np.nanmean(APlist_resin_45, axis=0)
mAP_pith_45 = np.nanmean(mAP_pith_45)
AP50_pith_45 = np.nanmean(AP50_pith_45)
APlist_pith_45 = np.nanmean(APlist_pith_45, axis=0)

## TP_FP_FN_group
TP_45 = np.array(np.sum(TP_45, axis=0))
FP_45 = np.array(np.sum(FP_45, axis=0))
FN_45 = np.array(np.sum(FN_45, axis=0))
### calculate sensitivity and precission
SEN_45 = TP_45/(TP_45+FN_45)
PREC_45 = TP_45/(TP_45+FP_45)
#print("SEN_45", SEN_45)
#print("PREC_45", PREC_45)

TP_ring_45 = np.array(np.sum(TP_ring_45, axis=0))
FP_ring_45 = np.array(np.sum(FP_ring_45, axis=0))
FN_ring_45 = np.array(np.sum(FN_ring_45, axis=0))
### calculate sensitivity and precission
SEN_ring_45 = TP_ring_45/(TP_ring_45+FN_ring_45)
PREC_ring_45 = TP_ring_45/(TP_ring_45+FP_ring_45)

TP_crack_45 = np.array(np.sum(TP_crack_45, axis=0))
FP_crack_45 = np.array(np.sum(FP_crack_45, axis=0))
FN_crack_45 = np.array(np.sum(FN_crack_45, axis=0))
### calculate sensitivity and precission
SEN_crack_45 = TP_crack_45/(TP_crack_45+FN_crack_45)
PREC_crack_45 = TP_crack_45/(TP_crack_45+FP_crack_45)

TP_resin_45 = np.array(np.sum(TP_resin_45, axis=0))
FP_resin_45 = np.array(np.sum(FP_resin_45, axis=0))
FN_resin_45 = np.array(np.sum(FN_resin_45, axis=0))
### calculate sensitivity and precission
SEN_resin_45 = TP_resin_45/(TP_resin_45+FN_resin_45)
PREC_resin_45 = TP_resin_45/(TP_resin_45+FP_resin_45)
#print("SEN_resin_45", SEN_resin_45)
TP_pith_45 = np.array(np.sum(TP_pith_45, axis=0))
FP_pith_45 = np.array(np.sum(FP_pith_45, axis=0))
FN_pith_45 = np.array(np.sum(FN_pith_45, axis=0))
### calculate sensitivity and precission
SEN_pith_45 = TP_pith_45/(TP_pith_45+FN_pith_45)
PREC_pith_45 = TP_pith_45/(TP_pith_45+FP_pith_45)

## IoU_group
IoU_45 = np.mean(IoU_45)
IoU_ring_45 = np.mean(IoU_ring_45)
IoU_crack_45 = np.mean(IoU_crack_45)
IoU_resin_45 = np.mean(IoU_resin_45)
IoU_pith_45 = np.mean(IoU_pith_45)

# Prec and recall for combined
#print('TPs_combined', TPs_combined)
TPs_combined = np.sum(TPs_combined)
FPs_combined = np.sum(FPs_combined)
FNs_combined = np.sum(FNs_combined)
#print('TPs_combined', TPs_combined)
SEN_combined = TPs_combined/(TPs_combined+FNs_combined)
PREC_combined = TPs_combined/(TPs_combined+FPs_combined)
IoU_combined_mask = np.mean(IoU_combined_mask)

iou_thresholds = np.arange(0.5, 1.0, 0.05) # for graph

print("mAP_crack", mAP_crack)
print("mAP_crack_45", mAP_crack_45)
print("mAP_crack_90", mAP_crack_90)
print("AP50_crack", AP50_crack)
print("AP50_crack_45", AP50_crack_45)
print("AP50_crack_90", AP50_crack_90)
#######################################################################
#Save output
#######################################################################
#get folder path and make folder
weight_path = args.weight
#print(run_path)
weight_path_split_1 = os.path.split(weight_path)
#print(run_split_1)
weight_name = weight_path_split_1[1]
#print('weight_name:', weight_name)
training_ID = os.path.split(weight_path_split_1[0])[1]
#print('run_ID:', run_ID)

model_eval_DIR = os.path.join(ROOT_DIR, 'samples/TreeRing/model_eval')
#print(model_eval_DIR)
training_eval_DIR = os.path.join(model_eval_DIR,training_ID)
weight_eval_DIR = os.path.join(training_eval_DIR, weight_name)

if not os.path.exists(training_eval_DIR): #check if it already exists and if not make it
    os.makedirs(training_eval_DIR)

if not os.path.exists(weight_eval_DIR): #check if it already exists and if not make it
    os.makedirs(weight_eval_DIR)

#save table
df = pd.DataFrame()

df['variables'] = ["mAP", "AP50", "mAP_ring", "AP50_ring", "mAP_crack", "AP50_crack", "mAP_resin", "AP50_resin", "mAP_pith", "AP50_pith",  "TP", "FP",
"FN", "SEN", "PREC", "TP_ring", "FP_ring", "FN_ring","SEN_ring", "PREC_ring", "TP_crack", "FP_crack", "FN_crack", "SEN_crack", "PREC_crack", "TP_resin",
"FP_resin", "FN_resin", "SEN_resin", "PREC_resin", "TP_pith", "FP_pith", "FN_pith", "SEN_pith", "PREC_pith","IoU", "IoU_ring", "IoU_crack", "IoU_resin",
"IoU_pith"] #names of all the variables

df['normal'] = [mAP, AP50, mAP_ring, AP50_ring, mAP_crack, AP50_crack, mAP_resin, AP50_resin, mAP_pith, AP50_pith, TP[0], FP[0], FN[0], SEN[0], PREC[0],
TP_ring[0], FP_ring[0], FN_ring[0], SEN_ring[0], PREC_ring[0], TP_crack[0], FP_crack[0], FN_crack[0], SEN_crack[0], PREC_crack[0], TP_resin[0], FP_resin[0],
FN_resin[0], SEN_resin[0], PREC_resin[0], TP_pith[0], FP_pith[0], FN_pith[0], SEN_pith[0], PREC_pith[0], IoU, IoU_ring, IoU_crack, IoU_resin, IoU_pith]#values for all the variables

df['90d'] = [mAP_90, AP50_90, mAP_ring_90, AP50_ring_90, mAP_crack_90, AP50_crack_90, mAP_resin_90, AP50_resin_90, mAP_pith_90, AP50_pith_90, TP_90[0],
FP_90[0], FN_90[0], SEN_90[0], PREC_90[0], TP_ring_90[0], FP_ring_90[0], FN_ring_90[0], SEN_ring_90[0], PREC_ring_90[0], TP_crack_90[0], FP_crack_90[0],
FN_crack_90[0], SEN_crack_90[0], PREC_crack_90[0], TP_resin_90[0], FP_resin_90[0], FN_resin_90[0], SEN_resin_90[0], PREC_resin_90[0], TP_pith_90[0],
FP_pith_90[0], FN_pith_90[0], SEN_pith_90[0], PREC_pith_90[0], IoU_90, IoU_ring_90, IoU_crack_90, IoU_resin_90, IoU_pith_90]#values for all the variables

df['45d'] = [mAP_45, AP50_45, mAP_ring_45, AP50_ring_45, mAP_crack_45, AP50_crack_45, mAP_resin_45, AP50_resin_45, mAP_pith_45, AP50_pith_45, TP_45[0],
FP_45[0], FN_45[0], SEN_45[0], PREC_45[0], TP_ring_45[0], FP_ring_45[0], FN_ring_45[0], SEN_ring_45[0], PREC_ring_45[0], TP_crack_45[0], FP_crack_45[0],
FN_crack_45[0], SEN_crack_45[0], PREC_crack_45[0], TP_resin_45[0], FP_resin_45[0], FN_resin_45[0], SEN_resin_45[0], PREC_resin_45[0], TP_pith_45[0],
FP_pith_45[0], FN_pith_45[0], SEN_pith_45[0], PREC_pith_45[0], IoU_45, IoU_ring_45, IoU_crack_45, IoU_resin_45, IoU_pith_45]#values for all the variables
print("df.shape", df.shape)
print("Data frame", df)
# combined
df['combined'] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan, TPs_combined, FPs_combined,
FNs_combined, SEN_combined, PREC_combined,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,
IoU_combined_mask, np.nan,np.nan,np.nan,]

# save the df as csv
df.to_csv(os.path.join(weight_eval_DIR, 'Evaluation_{}.csv'.format(weight_name)))

# save mAP graph
plt.plot(iou_thresholds, APlist, label= 'normal')
plt.plot(iou_thresholds, APlist_ring, label= 'ring')
plt.plot(iou_thresholds, APlist_crack, label= 'crack')
plt.plot(iou_thresholds, APlist_90, label= '90d')
plt.plot(iou_thresholds, APlist_45, label= '45d')
plt.ylabel('mAP')
plt.xlabel('IoU')
plt.legend()
#plt.ylim(bottom = 0, top = 1)
#plt.xlim(left = 0, right = 1)
plt.savefig(os.path.join(weight_eval_DIR, 'mAP_IoU_{}.jpg'.format(weight_name)))
plt.close()
##save graph data
df_mAP_graph = pd.DataFrame()
df_mAP_graph['mAP'] = APlist
df_mAP_graph['mAP_ring'] = APlist_ring
df_mAP_graph['mAP_crack'] = APlist_crack
df_mAP_graph['mAP_resin'] = APlist_resin
df_mAP_graph['mAP_pith'] = APlist_pith
df_mAP_graph['mAP_90'] = APlist_90
df_mAP_graph['mAP_ring_90'] = APlist_ring_90
df_mAP_graph['mAP_crack_90'] = APlist_crack_90
df_mAP_graph['mAP_resin_90'] = APlist_resin_90
df_mAP_graph['mAP_pith_90'] = APlist_pith_90
df_mAP_graph['mAP_45'] = APlist_45
df_mAP_graph['mAP_ring_45'] = APlist_ring_45
df_mAP_graph['mAP_crack_45'] = APlist_crack_45
df_mAP_graph['mAP_resin_45'] = APlist_resin_45
df_mAP_graph['mAP_pith_45'] = APlist_pith_45

df_mAP_graph['IoU_thresholds'] = iou_thresholds
df_mAP_graph.to_csv(os.path.join(weight_eval_DIR, 'mAP_IoU_graph_data_{}.csv'.format(weight_name)))

#save precission recall graph for mask and box together
plt.plot(SEN, PREC, label= 'general')
plt.plot(SEN_ring, PREC_ring, label= 'ring')
plt.plot(SEN_crack, PREC_crack, label= 'crack')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim(bottom = 0, top = 1)
plt.xlim(left = 0, right = 1)
plt.legend()
plt.savefig(os.path.join(weight_eval_DIR, 'PrecRec_{}.jpg'.format(weight_name)))
plt.close()
#save graph for all mask precision recal
plt.plot(SEN, PREC, label= 'general')
plt.plot(SEN_90, PREC_90, label= '90')
plt.plot(SEN_45, PREC_45, label= '45')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim(bottom = 0, top = 1)
plt.xlim(left = 0, right = 1)
plt.legend()
plt.savefig(os.path.join(weight_eval_DIR, 'PrecRecAllMasks_{}.jpg'.format(weight_name)))
plt.close()

##save precission recall data
df_PrecRec = pd.DataFrame()
df_PrecRec['Prec'] = PREC
df_PrecRec['Rec'] = SEN
df_PrecRec['Prec_ring'] = PREC_ring
df_PrecRec['Rec_ring'] = SEN_ring
df_PrecRec['Prec_crack'] = PREC_crack
df_PrecRec['Rec_crack'] = SEN_crack
df_PrecRec['Prec_resin'] = PREC_resin
df_PrecRec['Rec_resin'] = SEN_resin
df_PrecRec['Prec_pith'] = PREC_pith
df_PrecRec['Rec_pith'] = SEN_pith

df_PrecRec['Prec_90'] = PREC_90
df_PrecRec['Rec_90'] = SEN_90
df_PrecRec['Prec_ring_90'] = PREC_ring_90
df_PrecRec['Rec_ring_90'] = SEN_ring_90
df_PrecRec['Prec_crack_90'] = PREC_crack_90
df_PrecRec['Rec_crack_90'] = SEN_crack_90
df_PrecRec['Prec_resin_90'] = PREC_resin_90
df_PrecRec['Rec_resin_90'] = SEN_resin_90
df_PrecRec['Prec_pith_90'] = PREC_pith_90
df_PrecRec['Rec_pith_90'] = SEN_pith_90

df_PrecRec['Prec_45'] = PREC_45
df_PrecRec['Rec_45'] = SEN_45
df_PrecRec['Prec_ring_45'] = PREC_ring_45
df_PrecRec['Rec_ring_45'] = SEN_ring_45
df_PrecRec['Prec_crack_45'] = PREC_crack_45
df_PrecRec['Rec_crack_45'] = SEN_crack_45
df_PrecRec['Prec_resin_45'] = PREC_resin_45
df_PrecRec['Rec_resin_45'] = SEN_resin_45
df_PrecRec['Prec_pith_45'] = PREC_pith_45
df_PrecRec['Rec_pith_45'] = SEN_pith_45

df_PrecRec.to_csv(os.path.join(weight_eval_DIR, 'PrecRec_graph_data_{}.csv'.format(weight_name)))
