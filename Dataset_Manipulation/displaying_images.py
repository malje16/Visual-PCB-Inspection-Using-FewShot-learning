import json 
import cv2
import pycocotools.coco as tools
import coco_assistant as assistant
import os
import shutil
import skimage.io as io 
import matplotlib.pyplot as plt
import numpy as np 
import copy



def resize_img_percent(img, scale_percent=60):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def resize_img_sidelength(img, size=1000):
    if img.shape[1] > img.shape[0]:
        width = size
        scale_percent = int((width * 100) / img.shape[1])
        height = int(img.shape[0] * scale_percent / 100)
    else: 
        height = size
        scale_percent = int((height * 100) / img.shape[0])
        width = int(img.shape[1] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

img_dir="/home/maltenj/projects/dataset_manipulation/merged/images"


for name in img_names: 
    img = io.imread(os.path.join(img_dir, name))
    imgs.append(img)


ann = "/home/maltenj/projects/dataset_manipulation/manual_crop/annotations.json"
img = "/home/maltenj/projects/dataset_manipulation/manual_crop/images"

