import argparse
import dataclasses
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path

import cv2
import detectron2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from detectron2.config.config import CfgNode as CN






import os

from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch


from configs import thing_classes, Flags
from utils.utility import *
from dataset.process_data  import get_vinbigdata_dicts
from custom.evaluator import VinbigdataEvaluator
from custom.loss_hook import LossEvalHook
from custom.mapper import MyMapper, AlbumentationsMapper
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch




class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
        return build_detection_train_loader(
            cfg, mapper=AlbumentationsMapper(cfg, True, use_more_aug=True, cutmix_prob = 0.0, mixup_prob=0.0), sampler=sampler
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=AlbumentationsMapper(cfg, False)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # return PascalVOCDetectionEvaluator(dataset_name)  # not working
        # return COCOEvaluator(dataset_name, ("bbox",), False, output_dir=output_folder)
        return VinbigdataEvaluator(dataset_name, ("bbox",), False, output_dir=output_folder)

    def build_hooks(self):
        hooks = super(MyTrainer, self).build_hooks()
        cfg = self.cfg
        if len(cfg.DATASETS.TEST) > 0:
            loss_eval_hook = LossEvalHook(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                MyTrainer.build_test_loader(cfg, cfg.DATASETS.TEST[0]),
            )
            hooks.insert(-1, loss_eval_hook)

        return hooks



def main():
    setup_logger()

    flags_dict = {
        "debug": False,
        "outdir": "results/v9", 
        "imgdir_name": "vin_vig_256x256",
        "split_mode": "valid20",
        "iter": 10000,
        "roi_batch_size_per_image": 512,
        "eval_period": 10000,
        "lr_scheduler_name": "WarmupCosineLR",
        "checkpoint_interval":1000,
        "base_lr": 0.00025,
        "num_workers": 4,
        "aug_kwargs": {
            "HorizontalFlip": {"p": 0.5},
            "ShiftScaleRotate": {"scale_limit": 0.15, "rotate_limit": 10, "p": 0.5},
            "RandomBrightnessContrast": {"p": 0.3}
        }
    }

    # args = parse()
    print("torch", torch.__version__)
    flags = Flags().update(flags_dict)

    print("flags", flags)
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)

    flags_dict = dataclasses.asdict(flags)
    save_yaml(outdir / "flags.yaml", flags_dict)

    # --- Read data ---
    inputdir = Path("dataset/data")
    imgdir = inputdir / flags.imgdir_name

    # Read in the data CSV files
    train_df = pd.read_csv(inputdir / "train.csv")
    # train = train_df  # alias
    # sample_submission = pd.read_csv(datadir / 'sample_submission.csv')




    train_data_type = flags.train_data_type
    if flags.use_class14:
        thing_classes.append("No finding")

    #wether to use all data to train

    split_mode = flags.split_mode

    if split_mode == "all_train":
        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir, train_df, train_data_type, debug=debug, use_class14=flags.use_class14
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
    elif split_mode == "valid20":
        # To get number of data...
        n_dataset = len(
            get_vinbigdata_dicts(
                imgdir, train_df, train_data_type, debug=debug, use_class14=flags.use_class14
            )
        )
        n_train = int(n_dataset * 0.8)
        print("n_dataset", n_dataset, "n_train", n_train)
        rs = np.random.RandomState(flags.seed)
        inds = rs.permutation(n_dataset)
        train_inds, valid_inds = inds[:n_train], inds[n_train:]
        DatasetCatalog.register(
            "vinbigdata_train",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_df,
                train_data_type,
                debug=debug,
                target_indices=train_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
        DatasetCatalog.register(
            "vinbigdata_valid",
            lambda: get_vinbigdata_dicts(
                imgdir,
                train_df,
                train_data_type,
                debug=debug,
                target_indices=valid_inds,
                use_class14=flags.use_class14,
            ),
        )
        MetadataCatalog.get("vinbigdata_valid").set(thing_classes=thing_classes)
    else:
        raise ValueError(f"[ERROR] Unexpected value split_mode={split_mode}")





    cfg = get_cfg()
    cfg.aug_kwargs = CN(flags.aug_kwargs)  # pass aug_kwargs to cfg

    original_output_dir = cfg.OUTPUT_DIR
    cfg.OUTPUT_DIR = str(outdir)
    print(f"cfg.OUTPUT_DIR {original_output_dir} -> {cfg.OUTPUT_DIR}")

    config_name = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(config_name))
    cfg.DATASETS.TRAIN = ("vinbigdata_train",)
    if split_mode == "all_train":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = ("vinbigdata_valid",)
        cfg.TEST.EVAL_PERIOD = flags.eval_period

    cfg.DATALOADER.NUM_WORKERS = flags.num_workers
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_name)
    cfg.SOLVER.IMS_PER_BATCH = flags.ims_per_batch
    cfg.SOLVER.LR_SCHEDULER_NAME = flags.lr_scheduler_name
    cfg.SOLVER.BASE_LR = flags.base_lr  # pick a good LR
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.SOLVER.CHECKPOINT_PERIOD = flags.checkpoint_interval  # Small value=Frequent save need a lot of storage.
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
    # NOTE: this config means the number of classes,
    # but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    return trainer.train()


if __name__ == "__main__":
    main()
    
