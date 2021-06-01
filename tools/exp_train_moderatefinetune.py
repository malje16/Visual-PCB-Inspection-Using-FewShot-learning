"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in FsDet.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and
therefore may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use FsDet as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

from fsdet.config import get_cfg, set_global_cfg
from fsdet.engine import DefaultTrainer, default_argument_parser, default_setup
from fsdet.modeling import build_model

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

def My_train_aug(cfg):
    augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]
    
    augs.append(T.RandomExtent(scale_range=[0.90, 1], shift_range=[0.1, 0.1]))
    augs.append(T.RandomRotation(angle=[0, 90], sample_style="choice"))
    augs.append(T.RandomFlip(prob=0.5, horizontal=True, vertical=False))
    augs.append(T.RandomFlip(prob=0.5, horizontal=False, vertical=True))
    augs.append(T.RandomBrightness(0.9, 1.1))
    augs.append(T.RandomContrast(0.9, 1.1))
    augs.append(T.RandomSaturation(0.9, 1.1))

    return augs

def My_test_aug(cfg):
    augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING)]
    return augs



class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
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
    
    
def setup(args):
    """
    Create configs and perform basic setups.
    """
    register_my_datasets()
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(
            cfg.MODEL.WEIGHTS
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
