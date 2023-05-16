import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict

from util import box_ops, model_output_utils
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from util.model_output_utils import normalize_rel_maps

from .backbone import build_backbone
# from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegmRelMaps,
                           dice_loss, sigmoid_focal_loss, PostProcessSegm)
from .transformer import build_transformer

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

CLASSES_RNG = torch.arange(len(CLASSES))



def compute_fg_loss( relevance_map, target_seg, mse_critertion):
    pointwise_matrices = torch.mul(relevance_map, target_seg.float())
    fg_mse = mse_critertion(pointwise_matrices.float(), target_seg.float())
    # fg_mse_other = [mse_critertion(pointwise_matrices.float()[i], target_seg.float()[i]) for i in
    #                 range(pointwise_matrices.shape[0])]
    return fg_mse


def compute_bg_loss( relevance_map, target_seg, mse_critertion):
    neg_target_seg = torch.abs(
        (torch.ones_like(target_seg.float()) - target_seg.float()))  # this should neg the seg matrix
    pointwise_matrices = torch.mul(relevance_map, neg_target_seg)
    bg_mse = mse_critertion(pointwise_matrices, torch.zeros_like(pointwise_matrices))
    return bg_mse


def compute_relevance_loss(outputs_rel, targets_masks, fg_coeff, bg_coeff, reduction = 'mean'):
    # mse_criterion = torch.nn.MSELoss(reduction=reduction)
    mse_criterion_sum = torch.nn.MSELoss(reduction="sum")

    lamda_fg = fg_coeff
    lamga_bg = bg_coeff
    outputs_rel = outputs_rel.cuda()

    # fg_loss = compute_fg_loss(outputs_rel, targets_masks, mse_criterion)
    # bg_loss = compute_bg_loss(outputs_rel, targets_masks, mse_criterion)

    fg_loss_sum = compute_fg_loss(outputs_rel, targets_masks, mse_criterion_sum) / (targets_masks == 1).sum()
    bg_loss_sum = compute_bg_loss(outputs_rel, targets_masks, mse_criterion_sum) / (targets_masks == 0).sum()

    relevance_loss = lamda_fg * fg_loss_sum + lamga_bg * bg_loss_sum

    print(f"1 pixels : {(targets_masks ==1).sum()}")
    print(f"0 pixels : {(targets_masks == 0).sum()}")
    print(f"ratio : {targets_masks.mean()}")
    # print(f"bg loss : {bg_loss}")
    # print(f"fg loss : {fg_loss}")
    print(f"fg loss sum : {fg_loss_sum}")
    print(f"נg loss sum : {bg_loss_sum}")

    print(f"Rel loss: {relevance_loss}")


    return relevance_loss


def interpolate_and_resize(src_masks, target_masks):
    src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    src_masks = normalize_rel_maps(src_masks)
    return src_masks
def compute_rel_loss_from_map(outputs,idx, h, mask_generator, src_masks, targets, tgt_idx, w, tgt_img_num, tgt_mask_idx,
                              fg_coeff, bg_coeff, pred_class = None):
    src_masks = torch.reshape(src_masks, [src_masks.shape[0], src_masks.shape[1], h, w], )
    # new_masks = [torch.cat([mask_generator.get_panoptic_masks_no_thresholding(outputs,
    #                                                                           torch.tensor([mask_idx])) for mask_idx
    #                         in range((masks_amount))]) for i in range(batch_size)]
    # new_masks = torch.cat([torch.reshape(new_masks[i], [1, masks_amount, h, w]) for i in range(len(new_masks))])
    masks = [t["masks"] for t in targets]
    # for vis
    # TODO use valid to mask invalid areas due to padding in loss
    # this resizes all mask to (max_h, max_w) size
    target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    # target_masks = target_masks.to(src_masks)
    mask_generator.set_orig_rel(src_masks.detach())
    # upsample predictions to the target size
    src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                            mode="bilinear", align_corners=False)
    # Notice we norm twice.
    src_masks = normalize_rel_maps(src_masks)

    mask_generator.set_relevance(src_masks.detach())

    # # reshape masks from output
    # postprocessors = {'bbox': PostProcessRelMaps()}
    # postprocessors['segm'] = PostProcessSegmRelMaps()
    # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    # output_mask_results = postprocessors['bbox'](outputs, orig_target_sizes)
    # target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    # output_mask_results = postprocessors['segm'](output_mask_results, outputs, orig_target_sizes, target_sizes)

    # get the reshaped pred masks and original mask
    pred_masks = src_masks.squeeze(0).squeeze(0)
        # .squeeze(1)
    target_masks = target_masks[tgt_img_num][tgt_mask_idx].float()

    mask_generator.set_targets(target_masks.detach())

    # pred_boxes = [outputs["pred_boxes"][im][ind].float().cpu().unsqueeze(0) for im, ind in zip(idx[0], idx[1])]
    # pred_boxes = torch.cat(pred_boxes)
    # target_boxes = [targets[im]["boxes"][ind].float().cpu().unsqueeze(0) for im, ind in zip(tgt_idx[0], tgt_idx[1])]
    # target_boxes = torch.cat(target_boxes)
    # target_labels = [targets[im]["labels"][ind].float().cpu().unsqueeze(0) for im, ind in
    #                  zip(tgt_idx[0], tgt_idx[1])]
    # target_labels = torch.cat(target_labels)
    # pred_probs = [outputs["pred_logits"][im][ind].float().cpu().unsqueeze(0) for im, ind in zip(idx[0], idx[1])]
    # pred_probs = torch.cat(pred_probs)
    # loss = torch.tensor([self.compute_relevance_loss(pred_mask, target_mask) for pred_mask, target_mask in
    #                      zip(pred_masks, target_masks)]).sum() / num_boxes

    loss = compute_relevance_loss(pred_masks, target_masks, fg_coeff, bg_coeff)

    # label = targets[tgt_img_num]["labels"][tgt_mask_idx]
    # if(pred_class):
    #     correct_pred = pred_class.item() == label.item()
    # else:
    #     correct_pred = True
    # plt.scatter([label.item()], [loss.item()], c='blue' if correct_pred else 'red', edgecolors='black')

    del target_masks
    del pred_masks

    return loss
