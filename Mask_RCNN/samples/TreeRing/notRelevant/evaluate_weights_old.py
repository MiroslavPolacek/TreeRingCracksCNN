"""
Calculate a lits of mAP at IoU > 0.5 and for every epoch in the folder
--------------------------
Usage:

THIS TO TEST RUNS
conda activate TreeRingCNN &&
cd /Users/miroslav.polacek/Dropbox\ \(VBC\)/Group\ Folder\ Swarts/Research/CNNRings/Mask_RCNN/samples/TreeRing &&
python3 evaluate_weights_with_rotations.py  --dataset=/Users/miroslav.polacek/Dropbox\ \(VBC\)/Group\ Folder\ Swarts/Research/CNNRings/Mask_RCNN/datasets/treering  --weight=/Users/miroslav.polacek/Dropbox\ \(VBC\)/Group\ Folder\ Swarts/Research/CNNRings/Mask_RCNN/logs/JustToTest/mask_rcnn_treeringnewaug_0095.h5
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
import random
import math
import re
import time
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

#################################################################################
# Precision and recall for mask, first value of TP...should be for score of 0.5
#################################################################################
def TP_FP_NF_per_score_mask(gt_mask, pred_mask, scores, IoU_treshold):
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
        #print('mask_SR:', mask_SR.shape)
        mask_matrix = utils.compute_overlaps_masks(gt_mask, mask_SR)

    #for every score range callculate TP, ...append by the socre ranges
        # making binary numpy array with IoU treshold
        mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
        #print (mask_matrix_binary)


        #GT rings and predicted rigs
        gt_r = len(mask_matrix)
        pred_r = len(mask_matrix[0])

        #TP
        sum_truth = np.sum(mask_matrix_binary, axis=1)
        sum_truth_binary = np.where(sum_truth > 0, 1, 0)
        TP = np.sum(sum_truth_binary)
        TPs.append(TP)
        #print('TP:', TP)
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

#################################################################################
# Precision and recall for bboxes, first value of TP...should be for score of 0.5
#################################################################################
def TP_FP_NF_per_score_bbox(gt_bbox, pred_bbox, scores, IoU_treshold):
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
        bbox_SR = np.take(pred_bbox, score_ids, axis=0)
        #print('mask_SR:', mask_SR.shape)
        bbox_matrix = utils.compute_overlaps(gt_bbox, bbox_SR)

    #for every score range callculate TP, ...append by the socre ranges
        # making binary numpy array with IoU treshold
        bbox_matrix_binary = np.where(bbox_matrix > IoU_treshold, 1, 0)
        #print (bbox_matrix_binary)


        #GT rings and predicted rigs
        gt_r = len(bbox_matrix)
        pred_r = len(bbox_matrix[0])

        #TP
        sum_truth = np.sum(bbox_matrix_binary, axis=1)
        sum_truth_binary = np.where(sum_truth > 0, 1, 0)
        TP = np.sum(sum_truth_binary)
        TPs.append(TP)
        #print('TP:', TP)
        #FP
        sum_pred = np.sum(bbox_matrix_binary, axis=0)
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
    im, contours, hierarchy = cv2.findContours(uint8binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
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
# now calculate values for whole dataset
###########################################################################
# mAP, IoU
mAP = []
mask_IoU = []
bbox_IoU = []

mAP90 = []
mask_IoU90 = []
bbox_IoU90 = []

mAP45 = []
mask_IoU45 = []
bbox_IoU45 = []

# TPs, mask
TPs_mask = []
FPs_mask = []
FNs_mask = []

TPs90_mask = []
FPs90_mask = []
FNs90_mask = []

TPs45_mask = []
FPs45_mask = []
FNs45_mask = []

# TPs_mask_combined
TPs_combined = []
FPs_combined = []
FNs_combined = []

# TPs, bbox
TPs_bbox = []
FPs_bbox = []
FNs_bbox = []

TPs90_bbox = []
FPs90_bbox = []
FNs90_bbox = []

TPs45_bbox = []
FPs45_bbox = []
FNs45_bbox = []

# APlist for graph
APlist = []
APlist90 = []
APlist45 = []

# Iou for combined mask
IoU_combined_mask = []

for image_id in image_ids:
    ## Load the ground truth

    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    print('Evaluating image:', image_id)
    #print('image shape:', image.shape)
    imgheight = image.shape[0]

###### Detect image in normal orientation
    results = model.detect([image], verbose=0)
    r = results[0]
    if r['masks'].shape[-1] == 0:
        mAP.append(0)
        mask_IoU.append(0)
        bbox_IoU.append(0)
        TPs_mask.append(np.zeros(10))
        FPs_mask.append(np.zeros(10))
        FNs_mask.append(np.repeat(gt_mask.shape[-1], 10)) # this has to be the number of ground truth
        TPs_bbox.append(np.zeros(10))
        FPs_bbox.append(np.zeros(10))
        FNs_bbox.append(np.repeat(gt_mask.shape[-1], 10)) # this has to be the number of ground truth
        APlist.append(np.zeros(10))
        mask_normal = np.zeros(shape=(imgheight, imgheight))
        mask_normal = np.reshape(mask_normal, (1024,1024,1))
    else:
        ###calculate all the stuff
        mask_normal = r['masks'] # for the combined mask at the end
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        #print(r['scores'])
        #print(r['masks'].shape)
        mAP.append(ap)

        #compute mask IoU
        IoU_m = utils.compute_overlaps_masks(gt_mask, r['masks'])
        IoU_m = np.nan_to_num(np.mean(IoU_m)) #change nans to 0
        mask_IoU.append(IoU_m)

        #compute bbox IoU
        IoU_bbox = utils.compute_overlaps(gt_bbox,r['rois'])
        IoU_bbox = np.nan_to_num(np.mean(IoU_bbox))
        bbox_IoU.append(IoU_bbox)

        #compute TP, FP, FN for mask
        TP, FP, FN, score_range = TP_FP_NF_per_score_mask(gt_mask, r['masks'], r['scores'], IoU_treshold=0.3)
        #print(TP)
        #print(FP)
        #print(FN)
        TPs_mask.append(TP)
        FPs_mask.append(FP)
        FNs_mask.append(FN)

        #compute TP, FP, FN for bbox
        TP, FP, FN, score_range = TP_FP_NF_per_score_bbox(gt_bbox, r['rois'], r['scores'], IoU_treshold=0.3)
        #print(TP)
        #print(FP)
        #print(FN)
        TPs_bbox.append(TP)
        FPs_bbox.append(FP)
        FNs_bbox.append(FN)

        #APlist for graph
        ap = compute_ap_range_list(
            gt_bbox, gt_class_id, gt_mask,
            r['rois'], r['class_ids'], r['scores'], r['masks'],
            verbose=0)
        #print(ap)
        APlist.append(ap)


###### DETECT IMAGE 90degree
    ### rotate image, detect, rotate mask back
    image_90 = skimage.transform.rotate(image, 90, preserve_range=True).astype(np.uint8)
    #print('image90 shape:', image_90.shape)
    #plt.imshow(image_90)
    #plt.show()
    results = model.detect([image_90], verbose=0)
    r = results[0]
    if r['masks'].shape[-1] == 0:
        mAP90.append(0)
        mask_IoU90.append(0)
        bbox_IoU90.append(0)
        TPs90_mask.append(np.zeros(10))
        FPs90_mask.append(np.zeros(10))
        FNs90_mask.append(np.repeat(gt_mask.shape[-1], 10)) # this has to be the number of ground truth
        TPs90_bbox.append(np.zeros(10))
        FPs90_bbox.append(np.zeros(10))
        FNs90_bbox.append(np.repeat(gt_mask.shape[-1], 10)) # this has to be the number of ground truth
        APlist90.append(np.zeros(10))
        mask_90 = np.zeros(shape=(imgheight, imgheight))
        mask_90 = np.reshape(mask_90, (1024,1024,1))
    else:
        # rotate mask back
        mask_back = np.rot90(r['masks'], k=-1)
        mask_90 = mask_back # to combine at the end
        #plt.imshow(r['masks'][:,:,0])
        #plt.show()
        #plt.imshow(mask_back[:,:,0])
        #plt.show()

        # get the associated bbox
        #print(r['rois'].shape)
        #print(r['rois'][0])
        #plt.plot([r['rois'][0][1],r['rois'][0][3]])
        #plt.show()
        extracted_bboxes = utils.extract_bboxes(mask_back)

        ###calculate all the stuff
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            extracted_bboxes, r['class_ids'], r['scores'], mask_back,
            verbose=0)
        #print(r['scores'])
        #print(r['masks'].shape)
        mAP90.append(ap)

        #compute mask IoU
        IoU_m = utils.compute_overlaps_masks(gt_mask, mask_back)
        IoU_m = np.nan_to_num(np.mean(IoU_m)) #change nans to 0
        mask_IoU90.append(IoU_m)

        #compute bbox IoU
        IoU_bbox = utils.compute_overlaps(gt_bbox,extracted_bboxes)
        IoU_bbox = np.nan_to_num(np.mean(IoU_bbox))
        bbox_IoU90.append(IoU_bbox)

        #compute TP, FP, FN for mask
        TP, FP, FN, score_range = TP_FP_NF_per_score_mask(gt_mask, mask_back, r['scores'], IoU_treshold=0.3)
        #print(TP)
        #print(FP)
        #print(FN)
        TPs90_mask.append(TP)
        FPs90_mask.append(FP)
        FNs90_mask.append(FN)

        #compute TP, FP, FN for bbox
        TP, FP, FN, score_range = TP_FP_NF_per_score_bbox(gt_bbox, extracted_bboxes, r['scores'], IoU_treshold=0.3)
        #print(TP)
        #print(FP)
        #print(FN)
        TPs90_bbox.append(TP)
        FPs90_bbox.append(FP)
        FNs90_bbox.append(FN)

        #APlist for graph
        ap = compute_ap_range_list(
            gt_bbox, gt_class_id, gt_mask,
            extracted_bboxes, r['class_ids'], r['scores'], mask_back,
            verbose=0)
        #print(ap)
        APlist90.append(ap)

###### DETECT IMAGE 45degree
    ### rotate image, detect
    #print('img_shape:', image.shape)
    #plt.imshow(image)
    #plt.show()
    image_45 = skimage.transform.rotate(image, angle = 45, resize=True, preserve_range=True).astype(np.uint8)
    #print('img_45_shape:', image_45.shape)
    #plt.imshow(image_45)
    #plt.show()
    results = model.detect([image_45], verbose=0)
    r = results[0]
    # if mask is empty make all results zeroes
    if r['masks'].shape[-1] == 0:
        mAP45.append(0)
        mask_IoU45.append(0)
        bbox_IoU45.append(0)
        TPs45_mask.append(np.zeros(10))
        FPs45_mask.append(np.zeros(10))
        #print('FPs45_mask', FPs45_mask)
        #print('gt_mask:', gt_mask.shape[-1])
        FNs45_mask.append(np.repeat(gt_mask.shape[-1], 10)) # this has to be the number of ground truth
        #print('FNs45_mask', FNs45_mask)
        TPs45_bbox.append(np.zeros(10))
        FPs45_bbox.append(np.zeros(10))
        FNs45_bbox.append(np.repeat(gt_mask.shape[-1], 10)) # this has to be the number of ground truth
        APlist45.append(np.zeros(10))
        mask_45 = np.zeros(shape=(imgheight, imgheight))
        mask_45 = np.reshape(mask_45, (1024,1024,1))
    else:

        # rotate mask back
        #print('Rois_45:', r['rois'].shape)
        #print('Detected_mask_45:', r['masks'].shape)
        #plt.imshow(r['masks'][:,:,0])
        #plt.show()

        maskr2_back = skimage.transform.rotate(r['masks'], angle = -45, resize=False)
        #crop to the right size
        imgheight, imgwidth = image.shape[:2]
        imgheight2, imgwidth2 = maskr2_back.shape[:2]
        #print('img_shape:', image.shape)
        #print('img_45_shape:', maskr2_back.shape)
        to_crop = int((imgheight2 - imgheight)/2)

        mask_back = maskr2_back[to_crop: (to_crop+int(imgheight)), to_crop: (to_crop+int(imgheight))]
        mask_45 = mask_back #to combine at the end
        #plt.imshow(mask_back[:,:,0])
        #plt.show()

        # extract bounding boxes based on the masks
        extracted_bboxes = utils.extract_bboxes(mask_back)
        ###calculate all the stuff
        ap = utils.compute_ap_range(
            gt_bbox, gt_class_id, gt_mask,
            extracted_bboxes, r['class_ids'], r['scores'], mask_back,
            verbose=0)
        #print(r['scores'])
        #print(r['masks'].shape)
        mAP45.append(ap)

        #compute mask IoU
        IoU_m = utils.compute_overlaps_masks(gt_mask, mask_back)
        IoU_m = np.nan_to_num(np.mean(IoU_m)) #change nans to 0
        mask_IoU45.append(IoU_m)

        #compute bbox IoU
        IoU_bbox = utils.compute_overlaps(gt_bbox,extracted_bboxes)
        IoU_bbox = np.nan_to_num(np.mean(IoU_bbox))
        bbox_IoU45.append(IoU_bbox)

        #compute TP, FP, FN for mask
        TP, FP, FN, score_range = TP_FP_NF_per_score_mask(gt_mask, mask_back, r['scores'], IoU_treshold=0.3)
        #print(TP)
        #print(FP)
        #print(FN)
        TPs45_mask.append(TP)
        FPs45_mask.append(FP)
        FNs45_mask.append(FN)

        #compute TP, FP, FN for bbox
        TP, FP, FN, score_range = TP_FP_NF_per_score_bbox(gt_bbox, extracted_bboxes, r['scores'], IoU_treshold=0.3)
        #print(TP)
        #print(FP)
        #print(FN)
        TPs45_bbox.append(TP)
        FPs45_bbox.append(FP)
        FNs45_bbox.append(FN)

        #APlist for graph
        ap = compute_ap_range_list(
            gt_bbox, gt_class_id, gt_mask,
            extracted_bboxes, r['class_ids'], r['scores'], mask_back,
            verbose=0)
        #print(ap)
        APlist45.append(ap)

###### COMBINE ALL THE MASKS TO ONE AND CLACULATE IoU
    #normal flatten
    mask_normal_flat = np.zeros(shape=(imgheight, imgheight))
    nmasks = mask_normal.shape[2]
    for m in range(0,nmasks):
        mask_normal_flat = mask_normal_flat + mask_normal[:,:,m]
    #plt.imshow(mask_normal_flat)
    #plt.show()
    #90d flatten
    mask_90_flat = np.zeros(shape=(imgheight, imgheight))
    nmasks = mask_90.shape[2]
    for m in range(0,nmasks):
        mask_90_flat = mask_90_flat + mask_90[:,:,m]
    #45d flatten
    mask_45_flat = np.zeros(shape=(imgheight, imgheight))
    nmasks = mask_45.shape[2]
    for m in range(0,nmasks):
        mask_45_flat = mask_45_flat + mask_45[:,:,m]

    #combine to one
    combined_mask = mask_normal_flat + mask_90_flat + mask_45_flat
    #plt.imshow(combined_mask)
    #plt.show()

    #flatten ground truth mask
    gt_mask_flat = np.zeros(shape=(imgheight, imgheight))
    nmasks = gt_mask.shape[2]
    for m in range(0,nmasks):
        gt_mask_flat = gt_mask_flat + gt_mask[:,:,m]
    #calcumate IoU
    combined_mask_binary = np.where(combined_mask > 0, 1, 0)
    combined_mask_binary = np.reshape(combined_mask_binary, (1024,1024,1))

    #print('combined_mask_shape:', combined_mask_binary.shape)
    gt_mask_flat_binary = np.where(gt_mask_flat > 0, 1, 0)
    gt_mask_flat_binary = np.reshape(gt_mask_flat_binary, (1024,1024,1))
    #print('gt_mask_shape:', gt_mask_flat_binary.shape)
    IoU_combined_mask.append(utils.compute_overlaps_masks(gt_mask_flat_binary, combined_mask_binary))
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
    print(mask_matrix)
    # making binary numpy array with IoU treshold
    IoU_treshold = 0.3 # i set it less because combined mask is bigger and it does not matter i think
    mask_matrix_binary = np.where(mask_matrix > IoU_treshold, 1, 0)
    #print (mask_matrix_binary)


    #GT rings and predicted rigs
    gt_r = len(mask_matrix)
    pred_r = len(mask_matrix[0])

    #TP
    sum_truth = np.sum(mask_matrix_binary, axis=1)
    sum_truth_binary = np.where(sum_truth > 0, 1, 0)
    TP = np.sum(sum_truth_binary)
    TPs_combined.append(TP)
    #print('TP:', TP)
    #FP
    sum_pred = np.sum(mask_matrix_binary, axis=0)
    sum_pred_binary = np.where(sum_pred > 0, 1, 0)
    FP = pred_r - np.sum(sum_pred_binary)
    FPs_combined.append(FP)
    #print('FP:', FP)
    #FN
    FN = gt_r - TP
    FNs_combined.append(FN)
#calculate averages for all images
#0
mAP = np.mean(mAP)
mask_IoU = np.mean(mask_IoU)
bbox_IoU = np.mean(bbox_IoU)

# Prec, recall for mask
print('TPs_mask',TPs_mask)
TPs_mask = np.sum(TPs_mask, axis=0)
FPs_mask = np.sum(FPs_mask, axis=0)
FNs_mask = np.sum(FNs_mask, axis=0)

TPs_mask = np.array(TPs_mask)
FPs_mask = np.array(FPs_mask)
FNs_mask = np.array(FNs_mask)
SEN_mask = TPs_mask/(TPs_mask+FNs_mask)
PREC_mask = TPs_mask/(TPs_mask+FPs_mask)
#print('SEN_mask:', SEN_mask)
# Prec, recall for bbox
TPs_bbox = np.sum(TPs_bbox, axis=0)
FPs_bbox = np.sum(FPs_bbox, axis=0)
FNs_bbox = np.sum(FNs_bbox, axis=0)

#print('TP:', TPs)
#print('FP:', FPs)
#print('FN:', FNs)

TPs_bbox = np.array(TPs_bbox)
FPs_bbox = np.array(FPs_bbox)
FNs_bbox = np.array(FNs_bbox)
SEN_bbox = TPs_bbox/(TPs_bbox+FNs_bbox)
PREC_bbox = TPs_bbox/(TPs_bbox+FPs_bbox)

# Prec and recall for combined
print('TPs_combined', TPs_combined)
TPs_combined = np.sum(TPs_combined)
FPs_combined = np.sum(FPs_combined)
FNs_combined = np.sum(FNs_combined)
print('TPs_combined', TPs_combined)


SEN_combined = TPs_combined/(TPs_combined+FNs_combined)
PREC_combined = TPs_combined/(TPs_combined+FPs_combined)

#APlist for graph
APlist = np.mean(APlist, axis=0)
iou_thresholds = np.arange(0.5, 1.0, 0.05)

#90
mAP90 = np.mean(mAP90)
mask_IoU90 = np.mean(mask_IoU90)
bbox_IoU90 = np.mean(bbox_IoU90)

# Prec, recall for mask
#print('TPs90_mask:', TPs90_mask)
TPs90_mask = np.sum(TPs90_mask, axis=0)
FPs90_mask = np.sum(FPs90_mask, axis=0)
FNs90_mask = np.sum(FNs90_mask, axis=0)
#print('TPs90_mask_after:', TPs90_mask)
#print('TP:', TPs)
#print('FP:', FPs)
#print('FN:', FNs)

TPs90_mask = np.array(TPs90_mask)
FPs90_mask = np.array(FPs90_mask)
FNs90_mask = np.array(FNs90_mask)
SEN90_mask = TPs90_mask/(TPs90_mask+FNs90_mask)
PREC90_mask = TPs90_mask/(TPs90_mask+FPs90_mask)
#print('SEN90_mask:', SEN90_mask)
# Prec, recall for bbox
TPs90_bbox = np.sum(TPs90_bbox, axis=0)
FPs90_bbox = np.sum(FPs90_bbox, axis=0)
FNs90_bbox = np.sum(FNs90_bbox, axis=0)

#print('TP:', TPs)
#print('FP:', FPs)
#print('FN:', FNs)

TPs90_bbox = np.array(TPs90_bbox)
FPs90_bbox = np.array(FPs90_bbox)
FNs90_bbox = np.array(FNs90_bbox)
SEN90_bbox = TPs90_bbox/(TPs90_bbox+FNs90_bbox)
PREC90_bbox = TPs90_bbox/(TPs90_bbox+FPs90_bbox)

#APlist for graph
APlist90 = np.mean(APlist90, axis=0)


#45
#print('mAP45:', mAP45)
mAP45 = np.mean(mAP45)
mask_IoU45 = np.mean(mask_IoU45)
bbox_IoU45 = np.mean(bbox_IoU45)
#print('mAP45:', mAP45)
# Prec, recall for mask
#print('FNs45_mask:', FNs45_mask)
TPs45_mask = np.sum(TPs45_mask, axis=0)
FPs45_mask = np.sum(FPs45_mask, axis=0)
FNs45_mask = np.sum(FNs45_mask, axis=0)
#print('FNs45_mask:', FNs45_mask)
#print('TP:', TPs)
#print('FP:', FPs)
#print('FN:', FNs)

TPs45_mask = np.array(TPs45_mask)
FPs45_mask = np.array(FPs45_mask)
FNs45_mask = np.array(FNs45_mask)
SEN45_mask = TPs45_mask/(TPs45_mask+FNs45_mask)
PREC45_mask = TPs45_mask/(TPs45_mask+FPs45_mask)

# Prec, recall for bbox
TPs45_bbox = np.sum(TPs45_bbox, axis=0)
FPs45_bbox = np.sum(FPs45_bbox, axis=0)
FNs45_bbox = np.sum(FNs45_bbox, axis=0)

#print('TP:', TPs)
#print('FP:', FPs)
#print('FN:', FNs)

TPs45_bbox = np.array(TPs45_bbox)
FPs45_bbox = np.array(FPs45_bbox)
FNs45_bbox = np.array(FNs45_bbox)
SEN45_bbox = TPs45_bbox/(TPs45_bbox+FNs45_bbox)
PREC45_bbox = TPs45_bbox/(TPs45_bbox+FPs45_bbox)

#APlist for graph
APlist45 = np.mean(APlist45, axis=0)
# combined mask
IoU_combined_mask = np.mean(IoU_combined_mask)
#print('IoU_combined:', IoU_combined_mask)

#######################################################################
#Save output
#######################################################################
#get folder path and make folder
run_path = args.weight
#print(run_path)
run_split_1 = os.path.split(run_path)
#print(run_split_1)
weight_name = run_split_1[1]
#print('weight_name:', weight_name)
run_ID = os.path.split(run_split_1[0])[1]
#print('run_ID:', run_ID)

model_eval_DIR = os.path.join(ROOT_DIR, 'samples/TreeRing/model_eval')
#print(model_eval_DIR)
run_eval_DIR = os.path.join(model_eval_DIR,run_ID)
weight_eval_DIR = os.path.join(run_eval_DIR, weight_name)

if not os.path.exists(run_eval_DIR): #check if it already exists and if not make it
    os.makedirs(run_eval_DIR)

if not os.path.exists(weight_eval_DIR): #check if it already exists and if not make it
    os.makedirs(weight_eval_DIR)

#save table
df = pd.DataFrame()

df['variables'] = ['mAP', 'mask_IoU', 'bbox_IoU', 'SEN_mask', 'PREC_mask', 'TPs_mask', 'FPs_mask', 'FNs_mask', 'SEN_bbox', 'PREC_bbox', 'TPs_bbox', 'FPs_bbox', 'FNs_bbox'] #names of all the variables
df['normal'] = [mAP, mask_IoU, bbox_IoU, SEN_mask[0], PREC_mask[0], TPs_mask[0], FPs_mask[0], FNs_mask[0], SEN_bbox[0], PREC_bbox[0], TPs_bbox[0], FPs_bbox[0], FNs_bbox[0]]  #values for all the variables
df['90d'] = [mAP90, mask_IoU90, bbox_IoU90, SEN90_mask[0], PREC90_mask[0], TPs90_mask[0], FPs90_mask[0], FNs90_mask[0], SEN90_bbox[0], PREC90_bbox[0], TPs90_bbox[0], FPs90_bbox[0], FNs90_bbox[0]]  #values for all the variables
df['45d'] = [mAP45, mask_IoU45, bbox_IoU45, SEN45_mask[0], PREC45_mask[0], TPs45_mask[0], FPs45_mask[0], FNs45_mask[0], SEN45_bbox[0], PREC45_bbox[0], TPs45_bbox[0], FPs45_bbox[0], FNs45_bbox[0]]  #values for all the variables
# average all to a last colum
df['average'] = df.mean(numeric_only=True, axis=1)
df['combined'] = [np.nan, IoU_combined_mask, np.nan, SEN_combined, PREC_combined, TPs_combined, FPs_combined, FNs_combined, np.nan, np.nan, np.nan, np.nan, np.nan]
# save the df as csv
df.to_csv(os.path.join(weight_eval_DIR, 'Evaluation_{}.csv'.format(weight_name)))

# save mAP graph
plt.plot(iou_thresholds, APlist, label= 'normal')
plt.plot(iou_thresholds, APlist90, label= '90d')
plt.plot(iou_thresholds, APlist45, label= '45d')
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
df_mAP_graph['IoU_thresholds'] = iou_thresholds
df_mAP_graph.to_csv(os.path.join(weight_eval_DIR, 'mAP_IoU_graph_data_{}.csv'.format(weight_name)))

#save precission recall graph for mask and box together
plt.plot(SEN_mask, PREC_mask, label= 'mask')
plt.plot(SEN_bbox, PREC_bbox, label= 'bbox')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim(bottom = 0, top = 1)
plt.xlim(left = 0, right = 1)
plt.legend()
plt.savefig(os.path.join(weight_eval_DIR, 'PrecRec_{}.jpg'.format(weight_name)))
plt.close()
#save graph for all mask precision recal
plt.plot(SEN_mask, PREC_mask, label= 'mask')
plt.plot(SEN90_mask, PREC90_mask, label= 'mask90')
plt.plot(SEN45_mask, PREC45_mask, label= 'mask45')
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.ylim(bottom = 0, top = 1)
plt.xlim(left = 0, right = 1)
plt.legend()
plt.savefig(os.path.join(weight_eval_DIR, 'PrecRecAllMasks_{}.jpg'.format(weight_name)))
plt.close()
##save precission recall data
df_PrecRec = pd.DataFrame()
df_PrecRec['Prec_mask'] = PREC_mask
df_PrecRec['Rec_mask'] = SEN_mask
df_PrecRec['Prec_bbox'] = PREC_bbox
df_PrecRec['Rec_bbox'] = SEN_bbox
df_PrecRec['Prec90_mask'] = PREC90_mask
df_PrecRec['Rec90_mask'] = SEN90_mask
df_PrecRec['Prec90_bbox'] = PREC90_bbox
df_PrecRec['Rec90_bbox'] = SEN90_bbox
df_PrecRec['Prec45_mask'] = PREC45_mask
df_PrecRec['Rec45_mask'] = SEN45_mask
df_PrecRec['Prec45_bbox'] = PREC45_bbox
df_PrecRec['Rec45_bbox'] = SEN45_bbox
df_PrecRec.to_csv(os.path.join(weight_eval_DIR, 'PrecRec_graph_data_{}.csv'.format(weight_name)))

#######################################################################
# Print picture example for seprate tricky dataset
#######################################################################

# i will comment this now to speed this up
"""
Image_folder_path = '/Users/miroslav.polacek/Desktop/Rings_to_test'

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
        visualize_print.save_instances(image, r['rois'], r['masks'], r['class_ids'], dataset.class_names, out_image_path, r['scores'])
        #plt.figure(figsize=(30,30))
        #plt.axis('off')
        #plt.title(image_file)
        #plt.imshow(to_export)
        #plt.savefig(out_image_path)
        #plt.close()

"""
