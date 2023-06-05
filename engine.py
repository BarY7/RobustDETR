# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import copy
from typing import Dict
import math
import os
import sys
import gc
from typing import Iterable
import traceback
from contextlib import nullcontext


import cv2
import numpy as np
import requests
import torch
from torch.utils.tensorboard import SummaryWriter

from mask_generator import MaskGenerator \
    , rescale_bboxes, plot_results, plot_results_og
from models.matcher import HungarianMatcher
from models.segmentation import PostProcessSegm, PostProcessSegmOne
from PIL import Image

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from torch import nn
import matplotlib.pyplot as plt

from modules.modules.explainer import get_image_with_relevance, show_cam_on_image
from util import misc
from util.model_output_utils import otsu_thresh
from util.timer_utils import catchtime


def get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

def make_targets_single_class(targets: dict, class_id : int):
    # relies on single batch
    b = (targets[0]["labels"] == class_id)
    if b.any():
        targets[0]["labels"] = targets[0]["labels"][b.nonzero()[:, 0]]
        targets[0]["boxes"] = targets[0]["boxes"][b.nonzero()[:, 0]]
        targets[0]["area"] = targets[0]["area"][b.nonzero()[:, 0]]
        targets[0]["iscrowd"] = targets[0]["iscrowd"][b.nonzero()[:, 0]]
        if "masks" in targets[0]:  # rel map loss on
            targets[0]["masks"] = targets[0]["masks"][b.nonzero()[:, 0]]
    else:
        targets = None
    return targets


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, postprocessors: Dict[str, nn.Module],
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, orig_model, copied_matcher : HungarianMatcher, max_norm: float = 0,  output_dir = None, logger : SummaryWriter = None,
                    poison_orig_model = True, class_id = None
                    ):
    post_process_seg = PostProcessSegmOne()
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    len_set = len(data_loader)

    method: str = 'ours_no_lrp'
    print("using method {0} for visualization".format(method))

    count = 0
    save_interval = 1000
    memory_interval = 100

    dist = False
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        dist = True

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        count += 1

        print(f"I AM NUMBER {misc.get_rank()}")
        try:
            if class_id is not None:
                # relies on single batch
                targets = make_targets_single_class(targets, class_id)
                if targets is None:
                    continue


            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            mask_generator = MaskGenerator(model,
                                           criterion.weight_dict['loss_rel_maps'] if 'loss_rel_maps' in criterion.weight_dict else 0, dist=dist)

            if dist:
                cm = model.no_sync()
            else:
                cm = nullcontext()

            with cm:
                with catchtime('Fward') as t:
                    outputs = mask_generator.forward_and_update_feature_map_size(samples)
                if 'loss_rel_maps' in criterion.weight_dict and criterion.need_compute_rel_maps():
                    with catchtime('Compute rel maps for all masks') as t:
                        rel_masks = get_all_rel_masks(device, mask_generator, outputs, samples)
                        outputs["pred_rel_maps"] = rel_masks

                # if(epoch%2 == 0):
                #     print(5)
                # else:
                # poision!
                if poison_orig_model:
                    with catchtime('Poison') as t:
                        poison_with_orig_model(copied_matcher, orig_model, samples, targets)

                    # important because loss updates the gradient
                with catchtime('Compute loss') as t:
                    optimizer.zero_grad()
                    loss_dict = criterion(outputs, targets, mask_generator)

            # out of no sync
            weight_dict = criterion.weight_dict # verify required grad is false
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

            # we zero grad earlier - each ,ask accumaltes gradient so we can't use it here.
            # optimizer.zero_grad()

            # print(f"Printing grads before sync - should be different. process {misc.get_rank()}")
            # print(model.module.transformer.decoder.get_parameter('layers.0.multihead_attn.k_proj.weight').grad)
            losses.backward()
            # grads shuold be equal on all processes
            # print("Printing after - shuold be same")
            # print(model.module.transformer.decoder.get_parameter('layers.0.multihead_attn.k_proj.weight').grad)


            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            # metric_logger.write_to_tb(logger, "train", (epoch * len_set) + count - 1)

            # # debugging output

            # debugging output
            if count % save_interval == 0:
                if output_dir:
                    # orig_relevance = generate_relevance(orig_model, image_ten, index=class_name)
                    with catchtime('Visualize results') as t:
                        vis_results(count, epoch, mask_generator, output_dir, post_process_seg, samples, targets, "train")

            del outputs
            del samples
            del targets
            del losses
            # del mask_generator.gen.R_i_i
            # del mask_generator.gen.R_q_i
            # del mask_generator.gen.R_q_q
            del mask_generator.gen
            del mask_generator
            # torch.cuda.empty_cache()
        except BaseException as err:
            sys.stderr.write("\n")
            sys.stderr.write(f"Error found in iter {count} epoch {epoch}\n")
            checkpoint_path = f'{output_dir}/checkpoint_fail{epoch:04}_{count}.pth'
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)

            raise err


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    for key, val in ret_dict.items():
        if utils.is_main_process():
            logger.add_scalar(f"avg_{key}", val, epoch)
    return ret_dict


def poison_with_orig_model(copied_matcher, orig_model, samples, targets):
    orig_output = orig_model(samples)
    orig_output["pred_boxes"] = orig_output["pred_boxes"].detach()
    orig_outputs_without_aux = {k: v for k, v in orig_output.items() if k != 'aux_outputs'}
    # Retrieve the matching between the outputs of the last layer and the targets
    orig_indices = copied_matcher(orig_outputs_without_aux, targets)
    orig_src_idx = get_src_permutation_idx(orig_indices)
    just_batched_labels = orig_outputs_without_aux["pred_logits"].max(-1)[1]  # b x 100
    # for vis we dont want no objects
    just_batched_labels_no_none = orig_outputs_without_aux["pred_logits"][:, :, :-1].max(-1)[
        1]  # b x 100
    for o_img_i, l in enumerate(orig_indices):
        # targets[o_img_i]["o_pred_logits"] = just_batched_labels[o_img_i]
        targets[o_img_i]["labels_vis"] = copy.deepcopy(targets[o_img_i]["labels"])  # just temp!
        for o_i, t_i in zip(*l):
            targets[o_img_i]["boxes"][t_i] = orig_outputs_without_aux["pred_boxes"][o_img_i][o_i]
            # for the real labels we don't want no obj
            # Try without it
            # targets[o_img_i]["labels"][t_i] = just_batched_labels[o_img_i][o_i]
            targets[o_img_i]["labels_vis"][t_i] = just_batched_labels_no_none[o_img_i][o_i]


def get_all_rel_masks(device, mask_generator, outputs, samples):
    batch_size = len(samples.tensors)  # 1
    outputs_logits = outputs["pred_logits"]
    # logits_max_idx = outputs_logits.max(-1)[1] # b x 100
    total_masks = outputs_logits.shape[1]  # 100
    rel_masks = torch.zeros(batch_size, total_masks, 1, mask_generator.h, mask_generator.w).to(device)
    all_indexes = list(range(total_masks))
    masks_batch_size = 1
    batched_indexes = [all_indexes[k: k + masks_batch_size] for k in range(0, total_masks, masks_batch_size)]
    for img_idx in range(batch_size):
        for mask_idx_batch in batched_indexes:
            flattened_rel = mask_generator.compute_norm_rel_map_with_gen \
                (batch_size, img_idx, mask_idx_batch, outputs_logits, req_grad=False).detach()
            # index = logits_max_idx[img_idx, mask_idx_batch]).detach()
            rel_masks[img_idx][mask_idx_batch] = flattened_rel.reshape(len(mask_idx_batch), 1, mask_generator.h,
                                                                       mask_generator.w)
    return rel_masks


def visualize_results(count, epoch, mask_generator, output_dir, post_process_seg, samples, targets):
    images = samples.tensors
    relevance = mask_generator.get_relevance()
    target_masks_vis = mask_generator.get_targets()
    pred_probs_vis = mask_generator.get_probs().softmax(-1)
    target_boxes = mask_generator.get_tar_boxes()
    target_labels = mask_generator.get_labels()
    query_ids = mask_generator.get_query_ids()
    mask_count = 0
    for img_num in range(images.shape[0]):  # IMAGES NUM
        # target_boxes_vis = mask_generator.get_boxes()
        pred_boxes_vis = mask_generator.get_boxes()
        target_masks = targets[img_num]['masks']

        for mask_target_num in range(target_masks.shape[0]):  # MASK ID

            orig_rel = mask_generator.get_orig_rel()
            orig_size = targets[img_num]['orig_size']

            size_no_pad = targets[img_num]['size']
            resized_rel = post_process_seg(relevance[mask_count], size_no_pad, orig_size)
            resized_img = post_process_seg.inter_image_bilinear(images[img_num], size_no_pad, orig_size)
            resized_t_mask = post_process_seg(target_masks_vis[mask_count], size_no_pad,
                                              orig_size,
                                              thresh=False)
            resized_pred_box = rescale_bboxes(pred_boxes_vis[mask_count], orig_size)
            resized_target_box = rescale_bboxes(target_boxes[mask_count], orig_size)

            # resized_rel = otsu_thresh(resized_rel, )
            image = get_image_with_relevance(resized_img, torch.ones_like(resized_img))
            # img_raw = post_process_seg.inter(images[img_num], orig_size)
            # img_raw = get_image_with_relevance(img_raw, torch.ones_like(resized_img))
            fig = plt.figure()

            plot_results(fig, image, pred_probs_vis[mask_count], resized_pred_box, 1, title="Prediction")
            plot_results(fig, image, target_labels[mask_count], resized_target_box, 2, title="GT")

            new_vis = get_image_with_relevance(resized_img, resized_rel)
            new_vis_2 = show_cam_on_image(resized_img, resized_rel)
            vis = cv2.cvtColor(new_vis_2, cv2.COLOR_RGB2BGR)
            # new_vis_3 = show_cam_on_image(resized_img, new_vis)
            # old_vis = get_image_with_relevance(resized_img, orig_relevance[img_num])
            gt = get_image_with_relevance(resized_img,
                                          resized_t_mask
                                          )
            # h_img = cv2.hconcat([image, gt, new_vis, new_vis_2])

            ax2 = fig.add_subplot(3, 2, 3)
            ax2.title.set_text('RelMaps')

            plt.imshow(vis.astype(np.uint8))

            ax3 = fig.add_subplot(3, 2, 4)
            ax3.title.set_text('Segmentation')

            cmap = plt.cm.get_cmap('Blues').reversed()
            plt.imshow(orig_rel[mask_count].view(mask_generator.h, mask_generator.w).data.cpu().numpy(), cmap=cmap)

            plt.imshow(gt.astype(np.uint8))

            # did_save = cv2.imwrite(f'{output_dir}/train_samples/res_{count}_{img_num}_{mask_target_num}.jpg', h_img)

            # plt.imshow(h_img)
            # plt.show()
            plt.savefig(
                f'{output_dir}/train_samples/FIG_res_EPOCH_{epoch}_{count}_img{targets[img_num]["image_id"].item()}_{img_num}_query_{query_ids[mask_count]}_{mask_target_num}.jpg',
                dpi=300)
            mask_count += 1
            plt.close(fig)

            del resized_rel
            del resized_img
            del resized_t_mask
            del resized_pred_box
            del image
            del resized_target_box


# @torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, epoch, orig_model, copied_matcher, output_dir, logger = None,
             class_id = None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in postprocessors.keys())
    # if ("loss_rel_maps" in criterion.weight_dict):
    #     iou_types += ('segm')
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )
    # with torch.autograd.set_detect_anomaly(True):
    len_set = len(data_loader)
    count = 0
    pred_class = []
    real_class = []
    rel_loss_list = []

    # masks_rel_loss_fig = plt.figure(figsize=(16, 10))
    # plt.xticks(np.arange(0, 100, 3))
    # plt.xlabel("ClassID")
    # plt.ylabel("Rel Loss")
    # plt.scatter([i for i in range(CLASSES_RNG.shape[0])], [])

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        # print(f" Allocated {torch.cuda.memory_allocated()} , Max {torch.cuda.max_memory_allocated()}")
        # print(torch.cuda.memory_summary())
        # print(f"#### image id !!!!!!! {targets[0]['image_id']}")
        try:
            if class_id is not None:
                targets = make_targets_single_class(targets, class_id)
                if targets is None:
                    continue

            count += 1

            dist = False
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                dist = True

            mask_generator = MaskGenerator(model, criterion.weight_dict['loss_rel_maps'] if 'loss_rel_maps' in criterion.weight_dict else 0, dist=dist)

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with torch.enable_grad():
                outputs = mask_generator.forward_and_update_feature_map_size(samples)

            #BREAKS IF BATCH SIZE > 1
            outputs["pred_masks_dummy"] = torch.zeros(1, 100,*targets[0]["orig_size"])
            if 'loss_rel_maps' in criterion.weight_dict and criterion.need_compute_rel_maps():
                with catchtime('Compute rel maps for all masks') as t:
                    rel_masks = get_all_rel_masks(device, mask_generator, outputs, samples)

                    outputs["pred_rel_maps"] = rel_masks

            with torch.enable_grad():
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

            metric_logger.write_to_tb(logger, "test", (epoch * len_set) + count - 1)

            # rel_loss_list.append(loss_dict["loss_rel_maps"].item())
            # pred_class.append(outputs["pred_logits"])
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessors['bbox'](outputs, orig_target_sizes)

            if 'loss_rel_maps' in criterion.weight_dict:
                num_masks = sum([t["masks"].shape[0] for t in targets])
                outputs["pred_masks"] = mask_generator.get_orig_rel()
            if "pred_masks" in outputs and outputs["pred_masks"] is None:
                print("None pred masks!!!")
            if 'segm' in postprocessors.keys() and outputs["pred_masks"] is not None:
                target_sizes = torch.stack([t["size"] for t in targets], dim=0)
                results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes, mask_generator.get_src_idx())
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
            post_process_seg = PostProcessSegmOne()
            save_interval = 400
            memory_interval = 5


            # im = samples.tensors[0].cpu()
            # # keep only predictions with 0.7+ confidence
            # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            # keep = probas.max(-1).values > 0.9
            #
            # bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.cpu().permute(1,2,0).numpy().shape[:2])
            #
            # plot_results_og(im.permute(1,2,0).numpy(), probas[keep], bboxes_scaled)

            # RISKY TURNING ON AS IT KILLS GRAPHS WITH ZERO GRAD
            def bear_vis():
                # colors for visualization
                COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
                CLASSES = [
                    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
                    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
                    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
                    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
                    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush'
                ]
                gen = MaskGenerator(model).gen

                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.75

                if keep.nonzero().shape[0] <= 1:
                    return

                outputs['pred_boxes'] = outputs['pred_boxes'].cpu()

                url = 'http://images.cocodataset.org/val2017/000000120853.jpg'
                im = Image.open(requests.get(url, stream=True).raw)

                # convert boxes from [0; 1] to image scales
                bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep.cpu()], im.size)

                # use lists to store the outputs via up-values
                conv_features, enc_attn_weights, dec_attn_weights = [], [], []

                hooks = [
                    model.backbone[-2].register_forward_hook(
                        lambda self, input, output: conv_features.append(output)
                    ),
                    # model.transformer.encoder.layers[-1].self_attn.register_forward_hook(
                    #     lambda self, input, output: enc_attn_weights.append(output[1])
                    # ),
                    model.transformer.decoder.layers[-1].multihead_attn.register_forward_hook(
                        lambda self, input, output: dec_attn_weights.append(output[1])
                    ),
                ]

                for layer in model.transformer.encoder.layers:
                    hook = layer.self_attn.register_forward_hook(
                        lambda self, input, output: enc_attn_weights.append(output[1])
                    )
                    hooks.append(hook)

                model(samples)

                for hook in hooks:
                    hook.remove()

                # don't need the list anymore
                conv_features = conv_features[0]
                enc_attn_weights = enc_attn_weights[-1]
                dec_attn_weights = dec_attn_weights[0]

                # get the feature map shape
                h, w = conv_features['0'].tensors.shape[-2:]
                img_np = np.array(im).astype(float)

                fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
                for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
                    ax = ax_i[0]
                    cam = gen.generate_ours(samples, idx, use_lrp=False)
                    cam = (cam - cam.min()) / (cam.max() - cam.min())
                    cmap = plt.cm.get_cmap('Blues').reversed()
                    ax.imshow(cam.view(h, w).data.cpu().numpy(), cmap=cmap)
                    ax.axis('off')
                    ax.set_title(f'query id: {idx.item()}')
                    ax = ax_i[1]
                    ax.imshow(im)
                    ax.add_patch(plt.Rectangle((xmin.detach(), ymin.detach()), xmax.detach() - xmin.detach(),
                                               ymax.detach() - ymin.detach(),
                                               fill=False, color='blue', linewidth=3))
                    ax.axis('off')
                    ax.set_title(CLASSES[probas[idx].argmax()])
                image_id = None
                id_str = '' if image_id == None else image_id
                fig.tight_layout()
                plt.show()

            # bear_vis()

            # debugging output
            if count % save_interval == 0:
                vis_results(count, epoch, mask_generator, output_dir, post_process_seg, samples, targets, "test")

            del mask_generator
            # torch.cuda.memory_summary(device=None, abbreviated=False)
            # torch.cuda.empty_cache()
        except BaseException as err:
            print(err)
            raise err
            continue

    # logger.add_figure('High class loss', masks_rel_loss_fig, 0)

    # logger.add_figure('Masks num (x) , Loss (y)', masks_rel_loss_fig, 0)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    ret_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()} #not really ret
    for key, val in ret_dict.items():
        if utils.is_main_process():
            logger.add_scalar(f"avg_test_{key}", val, epoch)

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
            stat_names = ["AP", "AP50", "AP75", "AP_SMALL", "AP_MED", "AP_LARGE", "AR", "AR50", "AR75", "AR_SMALL", "AR_MED", "AR_LARGE"]
            for name, val in zip(stat_names,stats['coco_eval_bbox']):
                if utils.is_main_process():
                    logger.add_scalar(f'eval_{name}', val , epoch)

        # if 'segm' in postprocessors.keys():
        #     stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        #     stat_names = ["AP", "AP50", "AP75", "AP_SMALL", "AP_MED", "AP_LARGE", "AR", "AR50", "AR75", "AR_SMALL", "AR_MED", "AR_LARGE"]
        #     for name, val in zip(stat_names,stats['coco_eval_bbox']):
        #         if utils.is_main_process():
        #             logger.add_scalar(f'eval_{name}', val , epoch)
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


def vis_results(count, epoch, mask_generator, output_dir, post_process_seg, samples, targets, run_type):
    try:
        if output_dir:
            # orig_relevance = generate_relevance(orig_model, image_ten, index=class_name)
            images = samples.tensors
            relevance = mask_generator.get_relevance()
            target_masks_vis = mask_generator.get_targets()
            pred_probs_vis = mask_generator.get_probs().softmax(-1)
            target_boxes = mask_generator.get_tar_boxes()
            target_labels = mask_generator.get_labels()
            query_ids = mask_generator.get_query_ids()

            mask_count = 0
            for img_num in range(images.shape[0]):  # IMAGES NUM
                # target_boxes_vis = mask_generator.get_boxes()
                pred_boxes_vis = mask_generator.get_boxes()
                target_masks = targets[img_num]['masks']

                for mask_target_num in range(target_masks.shape[0]):  # MASK ID

                    orig_rel = mask_generator.get_orig_rel()
                    orig_size = targets[img_num]['orig_size']

                    size_no_pad = targets[img_num]['size']
                    resized_rel = post_process_seg(relevance[mask_count], size_no_pad, orig_size)
                    resized_img = post_process_seg.inter_image_bilinear(images[img_num], size_no_pad, orig_size)
                    resized_t_mask = post_process_seg(target_masks_vis[mask_count], size_no_pad,
                                                      orig_size,
                                                      thresh=False)
                    resized_pred_box = rescale_bboxes(pred_boxes_vis[mask_count], orig_size)
                    resized_target_box = rescale_bboxes(target_boxes[mask_count], orig_size)

                    # resized_rel = otsu_thresh(resized_rel, )
                    image = get_image_with_relevance(resized_img, torch.ones_like(resized_img))
                    # img_raw = post_process_seg.inter(images[img_num], orig_size)
                    # img_raw = get_image_with_relevance(img_raw, torch.ones_like(resized_img))
                    fig = plt.figure()

                    plot_results(fig, image, pred_probs_vis[mask_count], resized_pred_box, 1, title="Prediction")
                    plot_results(fig, image, target_labels[mask_count], resized_target_box, 2, title="GT")

                    new_vis = get_image_with_relevance(resized_img, resized_rel)
                    new_vis_2 = show_cam_on_image(resized_img, resized_rel)
                    vis = cv2.cvtColor(new_vis_2, cv2.COLOR_RGB2BGR)
                    # new_vis_3 = show_cam_on_image(resized_img, new_vis)
                    # old_vis = get_image_with_relevance(resized_img, orig_relevance[img_num])
                    gt = get_image_with_relevance(resized_img,
                                                  resized_t_mask
                                                  )
                    # h_img = cv2.hconcat([image, gt, new_vis, new_vis_2])

                    ax2 = fig.add_subplot(3, 2, 3)
                    ax2.title.set_text('RelMaps')

                    plt.imshow(vis.astype(np.uint8))

                    ax3 = fig.add_subplot(3, 2, 4)
                    ax3.title.set_text('Segmentation')

                    cmap = plt.cm.get_cmap('Blues').reversed()
                    plt.imshow(orig_rel[mask_count].view(mask_generator.h, mask_generator.w).data.cpu().numpy(),
                               cmap=cmap)

                    ax4 = fig.add_subplot(3, 2, 5)
                    ax4.title.set_text('RelMaps(Hila)')
                    plt.imshow(gt.astype(np.uint8))

                    # did_save = cv2.imwrite(f'{output_dir}/train_samples/res_{count}_{img_num}_{mask_target_num}.jpg', h_img)

                    # plt.imshow(h_img)
                    # plt.show()
                    plt.savefig(
                        f'{output_dir}/{run_type}_samples/FIG_res_EPOCH_{epoch}_{count}_{count}_img{targets[img_num]["image_id"].item()}_{img_num}_query_{query_ids[mask_count]}_{mask_target_num}.jpg',
                        dpi=300)
                    mask_count += 1
                    plt.close(fig)

                    del resized_rel
                    del resized_img
                    del resized_t_mask
                    del resized_pred_box
                    del image
                    del resized_target_box
    except BaseException as err:
        sys.stderr.write(str(err))
        traceback.print_exc()
        sys.stderr.write("LOGGING ERROR")
