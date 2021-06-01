#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Look at the output as coming from output = model(input)
visualise input[image] with predictions as in output[instance]
to run use the regular detectron2 run line. 
$ python3 -m tools.train_net_aug_output --num-gpus 1 --config-file configs/Danchell/merged_aug.py 

$ python3 -m tools.train_net_aug_output --num-gpus 1 --config-file configs/Danchell/merged_aug_tiny.py
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

logger = logging.getLogger("detectron2")

@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def My_train_aug(cfg):
    augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]

    augs.append(T.RandomExtent(scale_range=[0.4, 0.7], shift_range=[0.8, 0.8]))
    augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    augs.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    augs.append(T.RandomBrightness(0.8, 1.2))
    augs.append(T.RandomContrast(0.8, 1.2))
    augs.append(T.RandomSaturation(0.8, 1.2))

    return augs


def My_test_aug(cfg):
    augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]

    return augs


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        #evaluator_list.append(COCOEvaluator(dataset_name, 'bbox', True, output_folder))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        train_mapper = DatasetMapper(cfg, is_train=True, augmentations=My_train_aug(cfg))
        return build_detection_train_loader(cfg, mapper=train_mapper)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):

        test_mapper = DatasetMapper(cfg, is_train=False, augmentations=My_test_aug(cfg))
        return build_detection_test_loader(cfg, dataset_name, mapper=test_mapper)


def register_and_load(name, ann_file, img_dir):
    register_coco_instances(name, {}, ann_file , img_dir)
    load_coco_json(ann_file, img_dir, name)
    

def register_my_datasets():
    # registering merged dataset train and test in a 20%/80% split
    register_and_load('merged_train', "/home/maltenj/datasets/fics_pcb_merged/annotations/train.json", "/home/maltenj/datasets/fics_pcb_merged/train")
    register_and_load('merged_test', "/home/maltenj/datasets/fics_pcb_merged/annotations/test.json", "/home/maltenj/datasets/fics_pcb_merged/test")
    register_and_load('tiny_fics_pcb', "/home/maltenj/datasets/tiny_fics_pcb/annotations/tinyset.json", "/home/maltenj/datasets/tiny_fics_pcb/tinyset")
    
    register_and_load('train_2shot1', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_2shot1.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('train_2shot2', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_2shot2.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('train_2shot3', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_2shot3.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('train_10shot1', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_10shot1.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('train_10shot2', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_10shot2.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('train_10shot3', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_10shot3.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    
    register_and_load('test_2shot1', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_2shot1.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('test_2shot2', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_2shot2.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('test_2shot3', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_2shot3.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('test_10shot1', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_10shot1.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('test_10shot2', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_10shot2.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('test_10shot3', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_10shot3.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    
    register_and_load('val_2shot1', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_2shot1.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('val_2shot2', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_2shot2.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('val_2shot3', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/test_2shot3.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('val_10shot1', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_10shot1.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('val_10shot2', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_10shot2.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    register_and_load('val_10shot3', "/home/maltenj/datasets/HighResDanchellExperiment/annotations/train_10shot3.json", "/home/maltenj/datasets/HighResDanchellExperiment/test")
    

def setup():
    register_my_datasets()
    cfg = get_cfg()
    config_path = "../configs/Danchell_Experiment2/FsDet_2shot1.yaml"
    cfg.merge_from_file(config_path)
    #default_setup(cfg, args)
    return cfg


def main():
    cfg = setup()
    model = Trainer.build_model(cfg)

    MyCheckpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    MyCheckpointer.load(MyCheckpointer.get_checkpoint_file())

    backbone_counter = 0
    box_head_counter = 0
    proposal_generator_counter = 0

    for p in model.backbone.parameters(): # Count 120
        #print(p)
        backbone_counter += 1
        if backbone_counter > 115: 
            print(p)
    print("number of backbone (layers?) parameters: " + str(backbone_counter))

    for p in model.roi_heads.box_head.parameters(): # Count 4
        print(p)
        box_head_counter += 1
    print("number of box head (layers?) parameters: " + str(box_head_counter))

    for p in model.proposal_generator.parameters(): # Count 6
        #print(p)
        proposal_generator_counter += 1
    print("number of proposal generator (layers?) parameters: " + str(proposal_generator_counter))
    
    return
    

if __name__ == "__main__":
    main()
