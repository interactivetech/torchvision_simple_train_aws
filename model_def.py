
from typing import Any, Dict, Sequence, Union
import torch
import copy
from collections import defaultdict
from attrdict import AttrDict
from fcos import fcos_resnet50_fpn
from data import build_dataset, unwrap_collate_fn
from group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
import torchvision
from coco_eval import CocoEvaluator
from pycocotools import mask as coco_mask
import time
import datetime
from pycocotools.coco import COCO

import numpy as np
from determined.pytorch import (
    DataLoader,
    LRScheduler,
    PyTorchTrial,
    PyTorchTrialContext,
    MetricReducer,
)
from coco_eval import CocoEvaluator


TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]

def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)

class COCOReducer(MetricReducer):
    def __init__(self, base_ds, iou_types, cat_ids=[]):
        self.base_ds = base_ds
        self.iou_types = iou_types
        self.cat_ids = cat_ids
        self.reset()

    def reset(self):
        self.results = []

    def update(self, result):
        self.results.extend(result)

    def per_slot_reduce(self):
        return self.results

    def cross_slot_reduce(self, per_slot_metrics):
        coco_evaluator = CocoEvaluator(self.base_ds, self.iou_types)
        if len(self.cat_ids):
            for iou_type in self.iou_types:
                coco_evaluator.coco_eval[iou_type].params.catIds = self.cat_ids
        for results in per_slot_metrics:
            results_dict = {r[0]: r[1] for r in results}
            coco_evaluator.update(results_dict)

        for iou_type in coco_evaluator.iou_types:
            coco_eval = coco_evaluator.coco_eval[iou_type]
            coco_evaluator.eval_imgs[iou_type] = np.concatenate(
                coco_evaluator.eval_imgs[iou_type], 2
            )
            coco_eval.evalImgs = list(coco_evaluator.eval_imgs[iou_type].flatten())
            coco_eval.params.imgIds = list(coco_evaluator.img_ids)
            # We need to perform a deepcopy here since this dictionary can be modified in a
            # custom accumulate call and we don't want that to change coco_eval.params.
            # See https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py#L315.
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

        coco_stats = coco_evaluator.coco_eval["bbox"].stats.tolist()

        loss_dict = {}
        loss_dict["mAP"] = coco_stats[0]
        loss_dict["mAP_50"] = coco_stats[1]
        loss_dict["mAP_75"] = coco_stats[2]
        loss_dict["mAP_small"] = coco_stats[3]
        loss_dict["mAP_medium"] = coco_stats[4]
        loss_dict["mAP_large"] = coco_stats[5]
        return loss_dict
class ObjectDetectionTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.hparams = AttrDict(self.context.get_hparams())
        print(self.hparams) 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # define model
        model = fcos_resnet50_fpn(pretrained=False,num_classes=91)
        self.model = self.context.wrap_model(model)

        # wrap model

        # wrap optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov="nesterov",
        )
        self.optimizer = self.context.wrap_optimizer(optimizer)
        # Wrap LR Scheduler
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[16,22],
                                                            gamma=self.hparams.gamma)
        self.lr_scheduler = self.context.wrap_lr_scheduler(lr_scheduler,
                                                           step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)

    def build_training_data_loader(self) -> DataLoader:
        dataset, num_classes = build_dataset(image_set="train", args=AttrDict({
                                                'data_dir':self.hparams.data_dir,
                                                'backend':self.hparams.backend,
                                                'masks': self.hparams.masks,
                                                }))
        train_sampler = torch.utils.data.RandomSampler(dataset)
        group_ids = create_aspect_ratio_groups(dataset, k=3)
        train_batch_sampler = GroupedBatchSampler(train_sampler, 
                                                  group_ids, 
                                                  batch_size=self.context.get_per_slot_batch_size())
        data_loader = DataLoader(
                                 dataset, 
                                 batch_sampler=train_batch_sampler, 
                                 num_workers=self.hparams.num_workers, 
                                 collate_fn=unwrap_collate_fn
        )
        return data_loader

    def build_validation_data_loader(self) -> DataLoader:
        dataset_test, _ = build_dataset(image_set="val", args=AttrDict({
                                                'data_dir':self.hparams.data_dir,
                                                'backend':self.hparams.backend,
                                                'masks': self.hparams.masks,
                                                }))
        self.base_ds = get_coco_api_from_dataset(dataset_test)

        self.reducer = self.context.wrap_reducer(
            COCOReducer(self.base_ds,['bbox'],[]),
            for_training=False,
            for_validation=True
        )
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        data_loader_test = DataLoader(
                            dataset_test,
                            batch_size=self.context.get_per_slot_batch_size(),
                            sampler=test_sampler,
                            num_workers=0,
                            collate_fn=unwrap_collate_fn)
        return data_loader_test
    
    def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int) -> Dict[str, torch.Tensor]:
        batch_time_start = time.time()
        images, targets = batch
        images = list(image.to(self.device ,non_blocking=True) for image in images)
        targets = [{k: v.to(self.device ,non_blocking=True) for k, v in t.items()} for t in targets]
        loss_dict = self.model(images, targets)
        losses_reduced = sum(loss for loss in loss_dict.values())
        loss_value = losses_reduced.item()
        # if ind %10 == 0:
        print("loss: ",loss_value)
        print("losses_reduced: ",losses_reduced)
        self.context.backward(losses_reduced)
        self.context.step_optimizer(self.optimizer)
        total_batch_time = time.time() - batch_time_start
        total_batch_time_str = str(datetime.timedelta(seconds=int(total_batch_time)))
        print(f"Training time {total_batch_time_str}")
        return loss_dict
    
    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        images, targets = batch
        loss_dict, outputs = self.model(images, targets)
        print("Val Loss Dict: ",loss_dict)
        losses_reduced = sum(loss for loss in loss_dict.values())
        loss_value = losses_reduced.item()
        # if ind %10 == 0:
        print("val loss: ",loss_value)
        print("val losses_reduced: ",losses_reduced)
        # model_time = time.time()
        # outputs = self.model(images)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        # model_time_str = str(datetime.timedelta(seconds=int(model_time)))
        # print("Model Time: ",model_time_str)
        self.reducer.update(res)

        # evaluator_time = time.time()
        return loss_dict