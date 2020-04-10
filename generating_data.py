# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:43:41 2020

@author: Darshit.Purohit
"""

from random import random
import os
from glob import glob
import json
from tqdm import tqdm
from shutil import copyfile, copy
# Path to your images
image_paths= glob("Annotation dataset/*")
#Path to your annotations from VIA tool
annotation_file = 'queue_detection.json'
#clean up the annotations a little
annotations = json.load(open(annotation_file))
cleaned_annotations = {}
for k,v in annotations['_via_img_metadata'].items():
#    print(k)
    cleaned_annotations[v['filename']] = v
# create train and validation directories
#! mkdir procdata
#! mkdir procdata/val
#! mkdir procdata/train
train_annotations = {}
valid_annotations = {}
# 20% of images in validation folder
for img in tqdm(image_paths):
    x,y=img.split('\\')
#    print(x,'',y)
    # Image goes to Validation folder
    if random()<0.2:
        copy(img, 'procdata/val/')
        valid_annotations[y] = cleaned_annotations[y]
    else:
        copy(img, 'procdata/train/')
        train_annotations[y] = cleaned_annotations[y]
# put different annotations in different folders
with open('procdata/val/via_region_data(new).json', 'w') as fp:
    json.dump(valid_annotations, fp)
with open('procdata/train/via_region_data(new).json', 'w') as fp:
    json.dump(train_annotations, fp)