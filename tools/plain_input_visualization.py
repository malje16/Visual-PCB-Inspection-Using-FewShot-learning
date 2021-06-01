#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
The idea here is to visualize the input based on the config file set in the setup function. 
So, to run this file simply change the string in def setup and register the dataset and then run it. 

"""
import logging
import os
import sys
from collections import OrderedDict
import torch
import time

import cv2
import random

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup

import detectron2.utils.comm as comm
import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, DatasetMapper, build_detection_train_loader, build_detection_test_loader
import detectron2.data.transforms as T
from detectron2.engine import launch
from fsdet.evaluation import (
    COCOEvaluator, DatasetEvaluators, LVISEvaluator, PascalVOCDetectionEvaluator, verify_results)
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets import load_coco_json

from detectron2.utils.visualizer import Visualizer
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation import DatasetEvaluator, inference_on_dataset
import matplotlib.pyplot as plt 
from contextlib import contextmanager
import datetime

def My_train_aug(cfg):
    augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]

    augs.append(T.RandomExtent(scale_range=[0.4, 0.6], shift_range=[0.8, 0.8]))
    augs.append(T.RandomRotation(angle=[0, 90], sample_style="choice"))
    augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    augs.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    augs.append(T.RandomBrightness(0.8, 1.2))
    augs.append(T.RandomContrast(0.8,1.2))
    augs.append(T.RandomSaturation(0.8, 1.2))

    return augs

def register_and_load(name, ann_file, img_dir):
    register_coco_instances(name, {}, ann_file , img_dir)
    load_coco_json(ann_file, img_dir, name)
    

def register_my_datasets():
    register_and_load('merged_train', "/home/maltenj/datasets/fics_pcb_merged/annotations/train.json", "/home/maltenj/datasets/fics_pcb_merged/train")
    register_and_load('merged_test', "/home/maltenj/datasets/fics_pcb_merged/annotations/test.json", "/home/maltenj/datasets/fics_pcb_merged/test")
    register_and_load('tiny_fics_pcb', "/home/maltenj/datasets/tiny_fics_pcb/annotations/tinyset.json", "/home/maltenj/datasets/tiny_fics_pcb/tinyset")
    register_and_load('train_danchell', "/home/maltenj/datasets/HighResDanchell.v2-2048x1542.coco/annotations/instances_train.json", "/home/maltenj/datasets/HighResDanchell.v2-2048x1542.coco/train")
    register_and_load('valid_danchell', "/home/maltenj/datasets/HighResDanchell.v2-2048x1542.coco/annotations/instances_valid.json", "/home/maltenj/datasets/HighResDanchell.v2-2048x1542.coco/valid")
    register_and_load('test_danchell', "/home/maltenj/datasets/HighResDanchell.v2-2048x1542.coco/annotations/instances_test.json", "/home/maltenj/datasets/HighResDanchell.v2-2048x1542.coco/test")
    

def setup():
    config_path = "/home/maltenj/FrustratinglyFSOD/configs/Danchell_Experiment/Base0.yaml"

    register_my_datasets()
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg


def main():
    cfg = setup()
    train_mapper = DatasetMapper(cfg, is_train=True, augmentations=My_train_aug(cfg))
    data_loader = build_detection_train_loader(cfg, mapper=train_mapper)
    for data, iteration in zip(data_loader, range(1, 100)):
        for batch_element in data: 
            input_img = batch_element['image']
            input_instance = batch_element['instances']
            input_file_name = batch_element['file_name'].split("/")[-1]

            MyVisual = Visualizer(input_img.permute(1,2,0), instance_mode=1)
            VisImg_with_anns = MyVisual.overlay_instances(boxes=input_instance.gt_boxes)
            img_with_anns = cv2.cvtColor(VisImg_with_anns.get_image(), cv2.COLOR_BGR2RGB)
            plt.imshow(img_with_anns)
            print("imgsource: " + input_file_name)
            plt.show()

if __name__ == "__main__":
    main()
