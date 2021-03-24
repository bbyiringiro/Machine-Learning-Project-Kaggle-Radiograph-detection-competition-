"""
Referenced:
 - https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
 - https://www.kaggle.com/dhiiyaur/detectron-2-compare-models-augmentation/#data
"""
import copy,os
import logging
import numpy as np
import pandas as pd

import detectron2.data.transforms as T
import torch
from detectron2.data import detection_utils as utils
from pathlib import Path
import dataclasses
import cv2



def all_dicts():
    

    flags_dict = {
            "debug": False,
            "outdir": "results/v9", 
            "imgdir_name": "vin_vig_256x256",
            "split_mode": "valid20",
           
        }

    flags = Flags().update(flags_dict)

    # print("flags", flags)
    debug = flags.debug
    outdir = Path(flags.outdir)
    os.makedirs(str(outdir), exist_ok=True)

    flags_dict = dataclasses.asdict(flags)
    # save_yaml(outdir / "flags.yaml", flags_dict)

    # --- Read data ---
    inputdir = Path("dataset/data")
    imgdir = inputdir / flags.imgdir_name

    # Read in the data CSV files
    train_df = pd.read_csv(inputdir / "train.csv")

    return get_vinbigdata_dicts(imgdir, train_df, debug=True)



def mixup_image_and_boxes(one_dict,img1, all_dataset_dicts):  
    img1_d = one_dict
    img2_d = all_dataset_dicts[np.random.randint(0, len(all_dataset_dicts) - 1)] 
    # img1 = cv2.imread(one_dict["file_name"], cv2.IMREAD_COLOR).astype(np.float32)
    # img2 = cv2.imread(img2_d["file_name"], cv2.IMREAD_COLOR).astype(np.float32)
    img2 = utils.read_image(img2_d["file_name"], format="BGR")
    mixed_img = ((img1.astype(np.float32)+img2.astype(np.float32))/2)
    mixed_img_dict= img1_d
    mixed_img_dict['annotations']= img1_d['annotations']+img2_d['annotations']
    
    return mixed_img_dict, mixed_img
    # return one_dict, img1


def load_cutmix_image_and_boxes(one_dict,img1, all_dataset_dicts,imsize=256):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
        xc, yc = [int(np.random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [np.random.randint(0, len(all_dataset_dicts) - 1) for _ in range(4)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        img_dict_result = None
        for i, index in enumerate(indexes):
            img_d = all_dataset_dicts[index]
            if not img_dict_result:
                img_dict_result = img_d
                image = img1
            else:
                image = utils.read_image(img_d["file_name"], format="BGR") 
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
                new_bbox = np.clip([img_d['annotations'][j]['bbox'][0]+padw, img_d['annotations'][j]['bbox'][1]+padh, img_d['annotations'][j]['bbox'][2]+padw, img_d['annotations'][j]['bbox'][3]+padh],0, 2 * s)
                if new_bbox[0] == new_bbox[2] or new_bbox[1] == new_bbox[3]:
                    del img_d['annotations'][j]
                    break
                img_d['annotations'][j]['bbox'] = new_bbox
            img_dict_result['annotations'] +=img_d['annotations']


        return img_dict_result, result_image


class MyMapper:
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    def __init__(self, cfg, is_train: bool = True):
        aug_kwargs = cfg.aug_kwargs
        aug_list = [
            # T.Resize((800, 800)),
        ]
        if is_train:
            aug_list.extend([getattr(T, name)(**kwargs) for name, kwargs in aug_kwargs.items()])
        self.augmentations = T.AugmentationList(aug_list)
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        print(f"[MyDatasetMapper] Augmentations used in {mode}: {self.augmentations}")



    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     dataset_dict.pop("sem_seg_file_name", None)
        #     return dataset_dict

        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        annos = [
            utils.transform_instance_annotations(obj, transforms, image_shape)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict




"""
Referenced:
 - https://detectron2.readthedocs.io/en/latest/tutorials/data_loading.html
 - https://www.kaggle.com/dhiiyaur/detectron-2-compare-models-augmentation/#data
"""
import albumentations as A
import copy
import numpy as np

import torch
from detectron2.data import detection_utils as utils

from detectron2.data import DatasetCatalog, MetadataCatalog
import sys
sys.path.append('..')
from dataset.process_data  import get_vinbigdata_dicts

from configs import thing_classes, Flags






class AlbumentationsMapper:
    """Mapper which uses `albumentations` augmentations"""
    def __init__(self, cfg, is_train: bool = True, use_more_aug=False, cutmix_prob = 0.0, mixup_prob=0.0):
        aug_kwargs = cfg.aug_kwargs
        aug_list = [
        ]
        if is_train:
            aug_list.extend([getattr(A, name)(**kwargs) for name, kwargs in aug_kwargs.items()])
        self.transform = A.Compose(
            aug_list, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"])
        )
        self.is_train = is_train

        mode = "training" if is_train else "inference"
        # print(f"[AlbumentationsMapper] Augmentations used in {mode}: {self.transform}")
        if use_more_aug:
            self.use_more_aug = True
            self.all_dicts = all_dicts()
            self.cutmix_prob = cutmix_prob
            self.mixup_prob = mixup_prob

        else:
            self.use_more_aug = False
        


    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        image_shape = image.shape[:2]  # h, w


       # print(len(dataset_dict['annotations']))
        ########## Cutmix and mix up #####
        if self.use_more_aug and self.is_train:
     
            if np.random.random() < self.mixup_prob:
                res_dict, image= mixup_image_and_boxes(dataset_dict, image, self.all_dicts)
                dataset_dict = res_dict

            if np.random.random() < self.cutmix_prob:
                res_dict, image = load_cutmix_image_and_boxes(dataset_dict, image, self.all_dicts)
                dataset_dict = res_dict
                dataset_dict["annotations"] = dataset_dict["annotations"]
                ########
        

        prev_anno = dataset_dict["annotations"]
        bboxes = np.array([obj["bbox"] for obj in prev_anno], dtype=np.float32)
        # category_id = np.array([obj["category_id"] for obj in dataset_dict["annotations"]], dtype=np.int64)
        category_id = np.arange(len(dataset_dict["annotations"]))

       



        transformed = self.transform(image=image, bboxes=bboxes, category_ids=category_id)
        image = transformed["image"]
        annos = []
        for i, j in enumerate(transformed["category_ids"]):
            d = prev_anno[j]
            d["bbox"] = transformed["bboxes"][i]
            annos.append(d)
        

       

        #dataset_dict.pop("annotations", None)  # Remove unnecessary field.

        # # if not self.is_train:
        # #     # USER: Modify this if you want to keep them for some reason.
        # #     dataset_dict.pop("annotations", None)
        # #     dataset_dict.pop("sem_seg_file_name", None)
        # #     return dataset_dict

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        instances = utils.annotations_to_instances(annos, image_shape)
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict



###testing code ####
if __name__ == "__main__":

    a = AlbumentationsMapper(None, use_more_aug=True)
    from detectron2.utils.visualizer import Visualizer
    import matplotlib.pyplot as plt
    


    cols = 1
    rows = 1
    fig, ax = plt.subplots(rows, cols, figsize=(18, 18))


    for i in range(1):
        # ax =axes[0]
        
        dd = a(a.all_dicts[i])
        d, img = dd, dd['image']
        
        # print(d.keys())
        # visualizer = Visualizer(img[:, :, ::-1], metadata=None, scale=0.5)
        # out = visualizer.draw_dataset_dict(d)
        # # cv2_imshow(out.get_image()[:, :, ::-1])
        # cv2.imwrite(str(outdir / f"vinbigdata{index}.jpg"), out.get_image()[:, :, ::-1])
        plt.imshow(img[:, :, ::-1])
        # ax.set_title(f"{anom_ind}: image_id {anomaly_image_ids[index]}")
    plt.show()

