"""
Complex evaluation of weights.
Took about 40 min on a cpu on 2 val images
# Test on laptop
cd ~/Github/TreeRingCracksCNN/Mask_RCNN/samples/TreeRing &&
conda activate TreeRingCNN &&
bash eval_weight_realPost_simple_laptopDebug.sh
"""
#######################################################################
#Arguments
#######################################################################
import argparse

    # Parse command line arguments
parser = argparse.ArgumentParser(
        description='Calculate mAP for all epochs')

parser.add_argument('--TreeRingConf', required=True)

parser.add_argument('--dataset', required=True,
                    metavar="/path/to/ring/dataset/",
                    help='Directory to ring dataset')
parser.add_argument('--weight', required=True,
                    metavar="/path/to/weight/folder",
                    help="Path to weight file")
parser.add_argument('--path_out', required=True,
                    help="Path to save output")

args = parser.parse_args()

#######################################################################
# Prepare packages, models and images
#######################################################################

import os
import sys
import skimage
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from postprocessingCracksRings import sliding_window_detection, clean_up_mask, apply_mask, plot_lines

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize_print
import mrcnn.model as modellib
from mrcnn.model import log
# to allow for looping over multiple files via args
TreeRingConfName = args.TreeRingConf
#from samples.TreeRing import TreeRingConfName as TreeRing # this allows to parralelise over several config values at the sam time
TreeRing = __import__(TreeRingConfName, fromlist=[''])
# print GPU
from tensorflow.python.client import device_lib
print("LOCAL DIVICES", device_lib.list_local_devices())

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
# Set some paths
IMAGE_PATH_OUT = os.path.join(args.path_out, args.TreeRingConf)
if not os.path.exists(IMAGE_PATH_OUT): #check if it already exists and if not make it
    os.makedirs(IMAGE_PATH_OUT)
#################################################################################
# Evaluate detected masks (TP, FP, FN) in a range of score values
#################################################################################
def TP_FP_FN_IoU_per_score_mask(gt_mask, pred_mask, scores, IoU_threshold):

    #loop scores
    score_range = np.arange(0.5, 1.0, 0.05)
        #print(gt_r)
        #print(pred_r)

    TPs = []
    FPs = []
    FNs = []
    IoUs = []
    #print("GT_MASK_SHAPE", gt_mask.shape)
    for SR in score_range:
        #print("SR",SR)
        #print("scores", scores)
        score_ids = np.where(scores > SR)[0] #Ids for predictions above certain score threshold
        #print("score_ids", score_ids)
        #print("pred_mask.shape", pred_mask.shape)
        mask_SR = np.take(pred_mask, score_ids, axis=2)
        #print('mask_SR.shape:', mask_SR.shape)

        mask_matrix = utils.compute_overlaps_masks(gt_mask, mask_SR)
        #print("mask_matrix", mask_matrix)
        # calculate average IoU
        IoU_clean = mask_matrix[np.where(mask_matrix>0)]
        #print("IoU_clean", IoU_clean)
        IoU = np.nan_to_num(np.mean(IoU_clean))
        IoUs.append(IoU)
        #for every score range callculate TP, ...append by the socre ranges
        # making binary numpy array with IoU threshold
        mask_matrix_binary = np.where(mask_matrix > IoU_threshold, 1, 0)
        #print (mask_matrix_binary)

        #GT rings and predicted rigs
        #print("MASK MATRIX SHAPE", mask_matrix.shape)

        if mask_matrix.shape[0]==0:
            TPs.append(0)
            FPs.append(mask_SR.shape[-1]) # All predicted are false in this case
            FNs.append(0)
        else:
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

        #print('TPs:', TPs)
        #print('FPs:', FPs)
        #print('FNs:', FNs)
    #put together and sum up TP...per range

    return TPs, FPs, FNs, IoUs
#################################################################################
# Evaluate masks (TP, FP, FN) over a range of IoU thresholds
#################################################################################
def TP_FP_FN_IoU_matrices(gt_mask, pred_mask, scores, IoU_thresholds=None):
    """Compute TP, FP and FN over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    if IoU_thresholds is None:
        IoU_thresholds = np.arange(0, 1.0, 0.05)


    TPs = []
    FPs = []
    FNs = []
    IoUs = []
    for iou in IoU_thresholds:
        TP, FP, FN, IoU = TP_FP_FN_IoU_per_score_mask(gt_mask, pred_mask,
                                                            scores, IoU_threshold=iou)
        if iou == 0:
            IoUs = IoU
        TPs.append(TP)
        FPs.append(FP)
        FNs.append(FN)

    return TPs, FPs, FNs, IoUs
#######################################################################
# mAP graph
#######################################################################
def compute_ap_range_list(gt_box, gt_class_id, gt_mask,
                     pred_box, pred_class_id, pred_score, pred_mask,
                     IoU_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    if IoU_thresholds is None:
        IoU_thresholds = np.arange(0, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    APlist = []
    for iou_threshold in IoU_thresholds:
        ap, precisions, recalls, overlaps =\
            utils.compute_ap(gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask, iou_threshold=iou_threshold)
        APlist.append(ap)

        if verbose:
            print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))

    #print(APlist)
    return APlist
#################################################################################
# Evaluate detected masks (TP, FP, FN) for combined mask
#################################################################################
def TP_FP_FN_IoU_comb(gt_mask, pred_mask, IoU_thresholds=None):

    #loop scores
    if IoU_thresholds is None:
        IoU_thresholds = np.arange(0, 1.0, 0.05)
        #print(gt_r)
        #print(pred_r)

    TPs = []
    FPs = []
    FNs = []
    IoUs = []
    #print("GT_MASK_SHAPE", gt_mask.shape)
    for iou_threshold in IoU_thresholds:

        mask_matrix = utils.compute_overlaps_masks(gt_mask, pred_mask)
        #print("mask_matrix", mask_matrix)

        if iou_threshold == 0:
            # calculate average IoU
            IoU_clean = mask_matrix[np.where(mask_matrix>0)]
            #print("IoU_clean", IoU_clean)
            IoU = np.nan_to_num(np.mean(IoU_clean))
            IoUs.append(IoU)
        #for every score range callculate TP, ...append by the socre ranges
        # making binary numpy array with IoU threshold
        mask_matrix_binary = np.where(mask_matrix > iou_threshold, 1, 0)
        #print (mask_matrix_binary)

        #GT rings and predicted rigs
        #print("MASK MATRIX SHAPE", mask_matrix.shape)

        if mask_matrix.shape[0]==0:
            TPs.append(0)
            FPs.append(pred_mask.shape[-1]) # All predicted are false in this case
            FNs.append(0)
        else:
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

        #print('TPs:', TPs)
        #print('FPs:', FPs)
        #print('FNs:', FNs)
    #put together and sum up TP...per range

    return TPs, FPs, FNs, IoUs
##########################################################################
# Turn flat combined mask into array with layer per every mask
##########################################################################
def modify_flat_mask(mask):
    #### identify polygons with opencv
    uint8binary = mask.astype(np.uint8).copy()

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
##########################################################################
# Turn contours in binary mask
##########################################################################
def contours_to_binary(clean_contours,imheight,imwidth, debug=False):
    mask = np.zeros([imheight,imwidth, len(clean_contours)],
                    dtype=np.uint8)
    for i in range(len(clean_contours)):
        # separate x and y coords for contour
        x_list = []
        y_list = []
        for p in range(len(clean_contours[i])):
            [[x,y]] = clean_contours[i][p]
            x_list.append(x)
            y_list.append(y)
        mask = np.zeros([imheight,imwidth, len(clean_contours)],
                        dtype=np.uint8)
        r, c = skimage.draw.polygon(y_list, x_list)
        mask[r, c, i] = 1
    if debug==True:
        plt.imshow(mask[:,:,0])
        plt.show()
    return mask
###########################################################################
# Calculate TP_FP_NF_per_score_mask. General and per class.
###########################################################################
def TP_FP_FN_group(gt_mask, gt_class_id, pred_mask, pred_class_id, pred_scores, IoU_threshold=0.5):
    TP_FP_FN_general = []
    TP_FP_FN_names = ["score_range", "TP", "FP", "FN","TP_ring", "FP_ring", "FN_ring","TP_crack", "FP_crack", "FN_crack","TP_resin", "FP_resin", "FN_resin", "TP_pith", "FP_pith", "FN_pith"]
    # if no mask is detected
    if pred_mask.shape[-1] == 0:
        score_range = np.arange(0.5, 1.0, 0.05)
        FN = [gt_mask.shape[-1]]*10 # FN here is sum of all the ground truth masks
        TP_FP_FN_general = [score_range, [0]*10, [0]*10, FN]
        for i in range(1,5):
            TP = [0]*10
            FP = [0]*10
            FN = [gt_mask[:,:,gt_class_id==i].shape[-1]]*10
            TP_FP_FN_general.extend([TP, FP, FN])

    else:
        # for all classes
        TP, FP, FN, score_range = TP_FP_FN_per_score_mask(gt_mask, pred_mask, pred_scores, IoU_threshold=IoU_threshold)
        TP_FP_FN_general = [score_range, TP, FP, FN]
        for i in range(1,5):
            TP, FP, FN, score_range = TP_FP_FN_per_score_mask(gt_mask[:,:,gt_class_id==i], pred_mask[:,:,pred_class_id==i], pred_scores[pred_class_id==i], IoU_threshold=IoU_threshold)
            TP_FP_FN_general.extend([TP, FP, FN])

    return TP_FP_FN_general, TP_FP_FN_names
###########################################################################
# Calculate IoU general and per class.
###########################################################################
def IoU_per_score(gt_mask, gt_class_id, pred_mask):
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
#0 DEGREE ROTATION
## AP
APlist = []
## TP_FP_FN
TP = []
FP = []
FN = []
## IoU
IoU = []

#90 DEGREE ROTATION
## AP group _90
APlist_90 = []
## TP_FP_FN_group
TP_90 = []
FP_90 = []
FN_90 = []
## IoU_group
IoU_90 = []

#45 DEGREE ROTATION
APlist_45 = []
## TP_FP_FN_group
TP_45 = []
FP_45 = []
FN_45 = []
## IoU_group
IoU_45 = []

#COMBINED MASK
IoU_combined = []
TP_combined = []
FP_combined = []
FN_combined = []

## thresholds
iou_thresholds = np.arange(0, 1.0, 0.05)
score_range = np.arange(0.5, 1.0, 0.05)

# Main structure
for image_id in image_ids:

    ## Load the ground truth for the image
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config,
                               image_id, use_mini_mask=False)
    print('EVALUATING IMAGE:', image_id)
    imgheight = image.shape[0]

###### DETECT IMAGE in normal orientation
    results = model.detect([image], verbose=0)
    r = results[0]

    AP_L = compute_ap_range_list(gt_box=gt_bbox, gt_class_id=gt_class_id,
                        gt_mask=gt_mask, pred_box=r['rois'], pred_class_id=r['class_ids'],
                        pred_score=r['scores'], pred_mask=r['masks'],
                        IoU_thresholds=iou_thresholds, verbose=1)
    TPs, FPs, FNs, IoUs = TP_FP_FN_IoU_matrices(gt_mask, r['masks'], scores=r['scores'], IoU_thresholds=iou_thresholds)

    APlist.append(AP_L)
    TP.append(TPs)
    FP.append(FPs)
    FN.append(FN)
    IoU.append(IoUs)

    print("results so far:")
    print(APlist)
    print(TPs)
    print(FNs)
    print(IoU)

###### DETECT IMAGE 90degree

    ### rotate image, detect, rotate mask back
    image_90 = skimage.transform.rotate(image, 90, preserve_range=True).astype(np.uint8)
    results = model.detect([image_90], verbose=0)
    r = results[0]

    # rotate mask back
    mask_90_back = np.rot90(r['masks'], k=-1)
    # get the associated bbox
    extracted_bboxes_90 = utils.extract_bboxes(mask_90_back)

    ###calculate all the stuff
    AP_L_90 = compute_ap_range_list(gt_box=gt_bbox, gt_class_id=gt_class_id,
                        gt_mask=gt_mask, pred_box=extracted_bboxes_90, pred_class_id=r['class_ids'],
                        pred_score=r['scores'], pred_mask=mask_90_back,
                        IoU_thresholds=iou_thresholds, verbose=1)
    TPs_90, FPs_90, FNs_90, IoUs_90 = TP_FP_FN_IoU_matrices(gt_mask, mask_90_back, scores=r['scores'], IoU_thresholds=iou_thresholds)

    APlist_90.append(AP_L_90)
    TP_90.append(TPs_90)
    FP_90.append(FPs_90)
    FN_90.append(FN_90)
    IoU_90.append(IoUs_90)

    print("results so far 90:")
    print(APlist_90)
    print(TPs_90)
    print(FNs_90)
    print(IoU_90)

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

    # extract bounding boxes based from the masks
    extracted_bboxes_45 = utils.extract_bboxes(mask_45_back)

    ###calculate all the stuff
    AP_L_45 = compute_ap_range_list(gt_box=gt_bbox, gt_class_id=gt_class_id,
                        gt_mask=gt_mask, pred_box=extracted_bboxes_45, pred_class_id=r['class_ids'],
                        pred_score=r['scores'], pred_mask=mask_45_back,
                        IoU_thresholds=iou_thresholds, verbose=1)
    TPs_45, FPs_45, FNs_45, IoUs_45 = TP_FP_FN_IoU_matrices(gt_mask, mask_45_back, scores=r['scores'], IoU_thresholds=iou_thresholds)

    APlist_45.append(AP_L_45)
    TP_45.append(TPs_45)
    FP_45.append(FPs_45)
    FN_45.append(FN_45)
    IoU_45.append(IoUs_45)

    print("results so far 45:")
    print(APlist_45)
    print(TPs_45)
    print(FNs_45)
    print(IoU_45)

###### GET COMBINED MASK WITH POSTPROCESSING AND CLACULATE precission, recall and IoU
    TP_temp = []
    FP_temp = []
    FN_temp = []
    IoU_temp = []
    for SR in score_range:
        print("Looping through score range for combined mask, now at:", SR)
        detected_mask = sliding_window_detection(image = image, modelRing=model, min_score=SR, overlap = 0.75, cropUpandDown = 0)
        detected_mask_rings = detected_mask[:,:,0]
        #print("detected_mask_rings", detected_mask_rings.shape)
        #print("detected_mask_cracks", detected_mask_cracks.shape)
        clean_contours_rings = clean_up_mask(detected_mask_rings, is_ring=True)
        #print("clean_contours_rings", len(clean_contours_rings))

        combined_mask_binary = contours_to_binary(clean_contours_rings, imgheight, imgheight, debug=False)
        print('combined_mask_binary.shape', combined_mask_binary.shape)

        if False:
            # Ploting lines is moslty for debugging
            file_name = 'image'+ str(image_id)
            masked_image = image.astype(np.uint32).copy()
            masked_image = apply_mask(masked_image, detected_mask_rings, alpha=0.3)
            plot_lines(image=masked_image,file_name=file_name,
                        path_out=IMAGE_PATH_OUT, gt_masks=gt_mask[:,:,gt_class_id==1],
                        clean_contours = clean_contours_rings, debug=True)

        TPs, FPs, FNs, IoUs = TP_FP_FN_IoU_comb(gt_mask, combined_mask_binary, IoU_thresholds=iou_thresholds)

        TP_temp.append(TPs)
        FP_temp.append(FPs)
        FN_temp.append(FNs)
        IoU_temp.append(IoUs)
    # transpose the arrays so they have the same structure as the previous
    TP_T = np.array(TP_temp).T.tolist()
    FP_T = np.array(TP_temp).T.tolist()
    FN_T = np.array(TP_temp).T.tolist()
    IoU_T = []
    for i in IoU_temp:
        IoU_T.extend(i)

    IoU_combined.append(IoU_T)
    TP_combined.append(TP_T)
    FP_combined.append(FP_T)
    FN_combined.append(FN_T)

    print("TP_temp", TP_temp)
    print("IoU_temp", IoU_temp)

print("LOOP FINISHED, SOME RESULTS")
print(APlist, TP, FP)
print(IoU_combined, TP_combined)
#print("IoU_combined_mask",IoU_combined_mask)

#CALCULATE AVERAGES FOR ALL IMAGES
#0 DEGREE ROTATION
## AP
APlist_mean = np.nanmean(APlist, axis=0)
print("APlist_mean", APlist_mean)
