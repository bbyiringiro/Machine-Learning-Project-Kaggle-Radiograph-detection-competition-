
import dataclasses
import os
from dataclasses import dataclass
from distutils.util import strtobool
from pathlib import Path

import cv2
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
import matplotlib.pyplot as plt


import os



from configs import thing_classes, Flags
from utils.utility import *
from dataset.process_data  import get_vinbigdata_dicts


# from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator



flags_dict = {
    "debug": False,
    "outdir": "results/v9", 
    "imgdir_name": "vin_vig_256x256",
    "split_mode": "valid20",
}


flags = Flags().update(flags_dict)


# --- Read data ---
inputdir = Path("dataset/data")
datadir = inputdir 
imgdir = inputdir / flags.imgdir_name

# Read in the data CSV files
# train_df = pd.read_csv(datadir / "train.csv")
train_df = pd.read_csv(datadir / "train_wbf.csv")
train = train_df  # alias



train_data_type = flags.train_data_type
if flags.use_class14:
    thing_classes.append("No finding")

#wether to use all data to train

split_mode = flags.split_mode

if split_mode == "all_train":
    DatasetCatalog.register(
        "vinbigdata_train",
        lambda: get_vinbigdata_dicts(
            imgdir, train_df, train_data_type, debug=True, use_class14=flags.use_class14,  use_cache =True,
        ),
    )
    MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
elif split_mode == "valid20":
    # To get number of data...
    n_dataset = len(
        get_vinbigdata_dicts(
            imgdir, train_df, train_data_type, debug=True, use_class14=flags.use_class14, use_cache =True,
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
            debug=True,
            target_indices=train_inds,
            use_class14=flags.use_class14,
            use_cache =False
        ),
    )
    MetadataCatalog.get("vinbigdata_train").set(thing_classes=thing_classes)
    DatasetCatalog.register(
        "vinbigdata_valid",
        lambda: get_vinbigdata_dicts(
            imgdir,
            train_df,
            train_data_type,
            debug=True,
            target_indices=valid_inds,
            use_class14=flags.use_class14,
            use_cache =False

        ),
    )
    MetadataCatalog.get("vinbigdata_valid").set(thing_classes=thing_classes)
else:
    raise ValueError(f"[ERROR] Unexpected value split_mode={split_mode}")


# Read in the data CSV files

# print(train_df[train_df.image_id =='9a5094b2563a1ef3ff50dc5c7ff71345'].shape)
# train = train_df[train_df.image_id =='9a5094b2563a1ef3ff50dc5c7ff71345']



def mixup_image_and_boxes(index, get_dicts):  
        img1_d = dataset_dicts[index] 
        img2_d = dataset_dicts[np.random.randint(0, len(get_dicts) - 1)] 
        img1 = cv2.imread(img1_d["file_name"], cv2.IMREAD_COLOR).astype(np.float32)
        print(img1.shape)
        img2 = cv2.imread(img2_d["file_name"], cv2.IMREAD_COLOR).astype(np.float32)
        
        mixed_img = (img1+img2)/2
        mixed_img_dict= img1_d.copy()
        mixed_img_dict['annotations']= img1_d['annotations']+img2_d['annotations']
        
        return mixed_img_dict, mixed_img


def load_cutmix_image_and_boxes(index, get_dicts,imsize=256):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
        xc, yc = [int(np.random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [np.random.randint(0, len(get_dicts) - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        img_dict_result = None
        for i, index in enumerate(indexes):
            img_d = dataset_dicts[index]
            if not img_dict_result:
                img_dict_result = img_d
            image = cv2.imread(img_d["file_name"], cv2.IMREAD_COLOR).astype(np.float32)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            for j in range(len(img_d['annotations'])):
                img_d['annotations'][j]['bbox'] = [img_d['annotations'][j]['bbox'][0]+padw, img_d['annotations'][j]['bbox'][1]+padh, img_d['annotations'][j]['bbox'][2]+padw, img_d['annotations'][j]['bbox'][3]+padh]
            img_dict_result['annotations'] +=img_d['annotations']


        return img_dict_result, result_image



dataset_dicts = get_vinbigdata_dicts(imgdir, train, debug=True)


# Visualize data...
anomaly_image_ids = train.query("class_id != 14")["image_id"].unique()
train_meta = pd.read_csv(imgdir/"train_meta.csv")
anomaly_inds = np.argwhere(train_meta["image_id"].isin(anomaly_image_ids).values)[:, 0]

vinbigdata_metadata = MetadataCatalog.get("vinbigdata_train")

cols = 2
rows = 2
fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
axes = axes.flatten()

for index, anom_ind in enumerate(anomaly_inds[:cols * rows]):
    ax = axes[index]
    # print(anom_ind)
    
    i = np.random.randint(0, high=500)
    while(i not in anomaly_inds):
        i = np.random.randint(0, high=500)
    # print(i,anom_ind )
    anom_ind = i
    # d = dataset_dicts[anom_ind]
    d, img= load_cutmix_image_and_boxes(anom_ind, dataset_dicts)
    d, img= mixup_image_and_boxes(anom_ind, dataset_dicts)

    # img = cv2.imread(d["file_name"])
    # print(len(d['annotations']))
    # break
    # print(d['annotations'])
    visualizer = Visualizer(img[:, :, ::-1], metadata=vinbigdata_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    # cv2_imshow(out.get_image()[:, :, ::-1])
    #cv2.imwrite(str(outdir / f"vinbigdata{index}.jpg"), out.get_image()[:, :, ::-1])
    ax.imshow(out.get_image()[:, :, ::-1])
    ax.set_title(f"{anom_ind}: image_id {anomaly_image_ids[index]}")
plt.show()
