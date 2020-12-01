import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import scipy
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

from samples.TreeRing import TreeRing

# Load data
config = TreeRing.BalloonConfig()
BALLOON_DIR = '/Volumes/swarts/user/miroslav.polacek/CNN/LineOnlytif'
#BALLOON_DIR = '/Volumes/swarts/lab/DendroImages/ToAnnotate/'

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = TreeRing.BalloonDataset()
dataset.load_balloon(BALLOON_DIR,"val") #train or validation ,"train"

# Must call before using the dataset
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# exporting part
    
path_to_export = '/Users/miroslav.polacek/Desktop/LineAnotChecktif'

img_ids = dataset.image_ids
img_list = []
for img_id in img_ids:
    img_path = dataset.image_reference(img_id)
    img_name = os.path.basename(img_path)
    img_list.append(img_name)
    exported_list = os.listdir(path_to_export)
    print(len(exported_list))
    
    img = dataset.load_image(img_id)
    mask, class_ids = dataset.load_mask(img_id)
   
    bbox = utils.extract_bboxes(mask)
    to_export = visualize.display_instances(img, bbox, mask, class_ids, dataset.class_names)
    plt.figure(figsize=(30,30))
    plt.axis('off')
    plt.title(dataset.image_reference(img_id))
    plt.imshow(to_export)
    path_img_to_save = os.path.join(path_to_export,img_name)
    plt.savefig(path_img_to_save)
    plt.close()