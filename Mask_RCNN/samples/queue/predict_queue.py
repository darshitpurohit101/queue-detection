# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 11:02:45 2020

@author: Darshit.Purohit
"""

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import skimage

ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library
#from mrcnn import utils
#from mrcnn import visualize
#from mrcnn.visualize import display_images
import mrcnn.model as modellib
#from mrcnn.model import log

import queue_detect
def callme(path):
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    
    queue_WEIGHTS_PATH = "D:\Annotate\Mask_RCNN-master\Mask_RCNN-master\logs\queue20200303T1541\mask_rcnn_queue_0030.h5"
    
    config = queue_detect.BalloonConfig() 
#    queue_DIR = "D:\Annotate\procdata"
    
#    class InferenceConfig(config.__class__):
#        # Run detection on one image at a time
#        GPU_COUNT = 1
#        IMAGES_PER_GPU = 1
#    
#    config = InferenceConfig()
#    config.display()
#    
    DEVICE = "/cpu:0"
#    
#    def get_ax(rows=1, cols=1, size=16):
#        """Return a Matplotlib Axes array to be used in
#        all visualizations in the notebook. Provide a
#        central point to control graph sizes.
#        
#        Adjust the size attribute to control how big to render images
#        """
#        _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
#        return ax
#    
#    dataset = queue_detect.BalloonDataset()
#    dataset.load_balloon(queue_DIR, "train")
#    
#    dataset.prepare()
    
    #print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))
    
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
        
    weights_path = queue_WEIGHTS_PATH
    
    model.load_weights(weights_path, by_name=True)
    
    #image_id = random.choice(dataset.image_ids)
    #image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    #    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    #info = dataset.image_info[image_id]
    
    #print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
    #                                       dataset.image_reference(image_id)))
    ##im = input("Enter image path")
    img = skimage.io.imread(path)
    
    results = model.detect([img], verbose=1)
    
    
#    ax = get_ax(1)
    r = results[0]
#    a = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
#                                'queue', r['scores'], ax=ax,
#                                title="Predictions")
    print('ROI: ',r['rois'])
    #log("gt_class_id", gt_class_id)
    #log("gt_bbox", gt_bbox)
    #log("gt_mask", gt_mask)
    
    #def load_image(image_path):
    #    """Load the specified image and return a [H,W,3] Numpy array.
    #    """
    #    # Load image
    #    image = skimage.io.imread(image_path)
    #    # If grayscale. Convert to RGB for consistency.
    #    if image.ndim != 3:
    #        image = skimage.color.gray2rgb(image)
    #    # If has an alpha channel, remove it for consistency
    #    if image.shape[-1] == 4:
    #        image = image[..., :3]
    #    return image
    #
    #image = load_image("D:/Queue managment v1/first.jpg")
    #
    #results = model.detect([image], verbose=1)
    #
    ## Display results
    #ax = get_ax(1)
    #r = results[0]
    #a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
    #                            dataset.class_names, r['scores'], ax=ax,
    #                            title="Predictions")
    
path = "D:/Queue Managment v2/Input/input39.jpg"
callme(path)