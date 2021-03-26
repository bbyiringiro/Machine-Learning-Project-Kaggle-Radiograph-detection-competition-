import argparse
import dataclasses
import os

import sys
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path

import detectron2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm




import sys
sys.path.append('..')

from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor, DefaultTrainer, launch


from configs import thing_classes, Flags
from utils.utility import *
from dataset.process_data  import get_vinbigdata_dicts
from custom.evaluator import VinbigdataEvaluator
from custom.loss_hook import LossEvalHook
from custom.mapper import MyMapper, AlbumentationsMapper
# from detectron2.engine import DefaultPredictor, DefaultTrainer, launch

from detectron2.config.config import CfgNode as CN

from d2.train_net import Trainer
from d2.detr import DetrDatasetMapper, add_detr_config
from d2.detr import add_detr_config


# from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator

parser = argparse.ArgumentParser(
        description='Train')

parser.add_argument("resume_path")
parser.add_argument('--cutmix', default= -1, type=float)
parser.add_argument('--mixup', default= -1, type=float)

class MyTrainer(Trainer):
    # def __init__(self, cfg, _mydata_dicts):
    #     super().__init__(cfg)
    #     self.mydata_dicts = _mydata_dicts
    @classmethod
    def build_train_loader(cls, cfg, sampler=None):
#         mapper = DetrDatasetMapper(cfg, True)
        mapper=AlbumentationsMapper(cfg, True, use_more_aug=(cfg.cutmix > 0 or cfg.mixup > 0), cutmix_prob = cfg.cutmix, mixup_prob=cfg.mixup)
        return build_detection_train_loader(
            cfg, mapper= mapper , sampler=sampler
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



def main(args):
    setup_logger()
    
    resume_path = args.resume_path
    assert(os.path.exists(resume_path))
    assert('yaml' in resume_path)
    flags_dict = load_yaml(resume_path)
    print(flags_dict)
    

    # args = parse()
    print("torch", torch.__version__)
    flags = Flags().update(flags_dict)

    print("flags", flags)
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)

    flags_dict = dataclasses.asdict(flags)
    # save_yaml(outdir / "flags.yaml", flags_dict)

    # --- Read data ---
    inputdir = Path("../dataset/data")
    imgdir = inputdir / flags.imgdir_name

    # Read in the data CSV files
    train_df = pd.read_csv(inputdir / "train.csv")
    # print(train_df)
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

    add_detr_config(cfg)
    cfg.aug_kwargs = CN(flags.aug_kwargs)


    if not flags.is_new_config:
        assert(args.cutmix >=0 or args.mixup >=0)
        print(args.cutmix,  args.mixup , args.cutmix >=0 or args.mixup >=0)

        if args.cutmix >=0:
            c_prob = args.cutmix
        else:
            c_prob = 0
        if args.mixup >=0:
            m_prob = args.mixup
        else:
            m_prob = 0

        
    else: ## for new version of config read them directly
        c_prob = flags.cut_mix_prob
        m_prob = flags.mix_up_prob


    cfg.cutmix = c_prob
    cfg.mixup = m_prob
    cfg.merge_from_file("d2/configs/detr_256_6_6_torchvision.yaml")



    cfg.DATASETS.TRAIN = ("vinbigdata_train",)
    if split_mode == "all_train":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = ("vinbigdata_valid",)
        cfg.TEST.EVAL_PERIOD = flags.eval_period
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    cfg.DATALOADER.NUM_WORKERS = 8

    cfg.OUTPUT_DIR = str(outdir)

    cfg.MODEL.WEIGHTS = "converted_model.pth"
    



    cfg.MODEL.DETR.NUM_CLASSES = len(thing_classes)
    cfg.MODEL.DETR.NUM_OBJECT_QUERIES = 100
    cfg.MODEL.DETR.GIOU_WEIGHT = 2.0
    cfg.MODEL.DETR.L1_WEIGHT = 5.0
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)


    cfg.SOLVER.IMS_PER_BATCH = flags.ims_per_batch
    cfg.SOLVER.BASE_LR = flags.base_lr  
    cfg.SOLVER.MAX_ITER = flags.iter
    cfg.SOLVER.WARMUP_FACTOR = 1.0
    cfg.SOLVER.WARMUP_ITERS = 10
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.OPTIMIZER = 'ADAMW'
    cfg.SOLVER.ACKBONE_MULTIPLIER = 0.1
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = 'full_model'
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.1
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.SOLVER.CHECKPOINT_PERIOD = flags.checkpoint_interval  # Small value=Frequent save need a lot of storage.
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = flags.roi_batch_size_per_image
# NOTE: this config means the number of classes,
# but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = DefaultTrainer(cfg) 
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
