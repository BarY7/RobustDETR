# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
from typing import Dict
import math
import os
import sys
from typing import Iterable

import torch
from mask_generator import MaskGenerator
from models.segmentation import PostProcessSegm

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from torch import nn
import matplotlib.pyplot as plt


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors: Dict[str, nn.Module],
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    mask_generator = MaskGenerator(model)
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    method: str = 'ours_no_lrp'
    print("using method {0} for visualization".format(method))

    # temp_count_4480 = 0
    # iterator = iter(data_loader)
    # consume(iterator, 4480)
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # temp_count_4480 += samples.tensors.shape[0]
        # if(temp_count_4480 < 4480):
        #     continue
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        batch_size = len(samples.tensors)

        # outputs = model(samples)

        # update in mask generator so rel maps loss can access this
        outputs = mask_generator.forward_and_update_feature_map_size(samples)

        # SET_FEATURE_MAP_SIZE = True

        # masks_amount = outputs['pred_logits'].shape[1]

        # h, w = mask_generator.update_feature_map_size(outputs, samples)

        # x = mask_generator.get_panoptic_masks(outputs, samples, targets, "ours_no_lrp")
        # masks = torch.cat([mask_generator.get_panoptic_masks_no_thresholding(get_one_output_from_batch(outputs, i), utils.NestedTensor(*samples.decompose_single_item(i)), targets[i], method) for i in range(batch_size)])

        # new_masks = [torch.cat([mask_generator.get_panoptic_masks_no_thresholding(samples.tensors[i].unsqueeze(0),
        #                                                                           torch.tensor([mask_idx])) for mask_idx
        #                         in range((masks_amount))]) for i in range(batch_size)]
        # new_masks = torch.cat([torch.reshape(new_masks[i], [1, masks_amount, h, w]) for i in range(len(new_masks))])

        # outputs["pred_masks"] = new_masks

        # # reshape masks
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](feature_map_relevancy, orig_target_sizes)
        # # if 'segm' in postprocessors.keys():
        # postprocessors['segm'] = PostProcessSegm()
        # target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        # results = postprocessors['segm'](results, feature_map_relevancy, orig_target_sizes, target_sizes)

        loss_dict = criterion(outputs, targets, mask_generator)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        torch.cuda.memory_summary(device=None, abbreviated=False)
        # torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, is_rel_maps: bool = False):
    model.eval()
    criterion.eval()

    mask_generator = MaskGenerator(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    if (is_rel_maps):
        iou_types += 'segm'
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = mask_generator.forward_and_update_feature_map_size(samples)
        loss_dict = criterion(outputs, targets, mask_generator=mask_generator)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        # torch.cuda.memory_summary(device=None, abbreviated=False)
        # torch.cuda.empty_cache()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
