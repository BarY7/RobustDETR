import torch
from modules.modules.ExplanationGenerator import Generator, GeneratorAlbationNoAgg
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from util.model_output_utils import normalize_rel_maps



def plot_results_og(pil_img, prob, boxes):
    # plt.figure(figsize=(16,10))
    # pil_img = (pil_img - pil_img.min()) / (pil_img.max() - pil_img.min())
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.tensor(b)

def rescale_bboxes(out_bbox, size):
    img_h, img_w = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(fig,pil_img, p, box, idx, title=''):
    ax1 = fig.add_subplot(3, 2, idx)
    plt.imshow(pil_img.astype(np.uint32))
    ax = plt.gca()
    colors = COLORS * 100

    (xmin, ymin, xmax, ymax) = box.numpy().astype(np.uint32)


    if (xmin < 0 or ymin < 0):
        print("OVERFLOW")
    if (xmax > pil_img.shape[1] or ymax > pil_img.shape[0]):
        print("OVERFLOW")

    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                               fill=False, color=[0.494, 0.684, 0.556], linewidth=1,))

    # ax.add_patch(plt.Rectangle((xmin, ymin), xmax, ymax, fill=False, color=[0.994, 0.184, 0.556], linewidth=3, ))
    if title != "GT":
        cl = p[:-1].argmax()
        pres = p[cl]
        ind = cl
    else:
        cl = p
        pres = 10
        ind = int(cl.item())

    try:
        text = f'{CLASSES[ind]}: {pres:0.2f}'
    except BaseException:
        text = f'NO OBJ: {pres:0.2f}'

    ax1.title.set_text(f"{title} {text}")

    # ax.text(xmin, ymin, text, fontsize=8,
    #         bbox=dict(facecolor='yellow', alpha=0.3)
    #         )

    plt.axis('off')
    # plt.show()


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

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

class MaskGenerator:
    def __init__(self, model, weight_coef):
        self.gen = Generator(model)
        # self.abl = GeneratorAlbationNoAgg(model)
        self.model = model
        self.h = None
        self.w = None
        self.weight_coef = weight_coef

        self.src_idx = None

        # for debugging purpose, only value
        self.relevance = None
        self.target_masks = None
        self.pred_boxes = None # vis
        self.probs = None #vis
        self.tar_labels = None
        self.tar_boxes = None

        self.orig_rel = None
        self.query_ids = None

    def forward_and_update_feature_map_size(self,  samples):
        # keep only predictions with 0.8+ confidence
        # probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        # keep = probas.max(-1).values > 0.5

        # ########### for visualizations
        # boxes = outputs['pred_boxes'].cpu()
        # im = samples.tensors[0].permute(1, 2, 0).data.cpu().numpy()
        # im = (im - im.min()) / (im.max() - im.min())
        # im = np.uint8(im * 255)
        # im = Image.fromarray(im)
        # # im = T.ToPILImage()(samples.tensors[0])
        # # convert boxes from [0; 1] to image scales
        # bboxes_scaled = rescale_bboxes(boxes[0, keep.cpu()], im.size)
        # ############ for visualizations


        # if keep.nonzero().shape[0] <= 1:
        #     print("no segmentation")

        # use lists to store the outputs via up-values
        vis_shape, target_shape = [], []

        hooks = [
            self.model.transformer.register_forward_hook(
                lambda self, input, output: vis_shape.append(output[1])
            ),
            self.model.backbone[-2].register_forward_hook(
                lambda self, input, output: target_shape.append(output)
            )
        ]

        outputs = self.model(samples)

        for hook in hooks:
            hook.remove()

        h, w = vis_shape[0].shape[-2:]
        self.h, self.w = h, w

        del vis_shape
        del target_shape

        return outputs

    def is_train_mode(self):
        return self.model.training
    def get_weight_coef(self):
        return self.weight_coef
    def get_panoptic_masks(self, outputs, samples,targets,method):
                # propagate through the model

        # keep only predictions with 0.8+ confidence
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5

        ########### for visualizations
        boxes = outputs['pred_boxes'].cpu()
        im = samples.tensors[0].permute(1, 2, 0).data.cpu().numpy()
        im = (im - im.min()) / (im.max() - im.min())
        im = np.uint8(im * 255)
        im = Image.fromarray(im)
        # im = T.ToPILImage()(samples.tensors[0])
        # convert boxes from [0; 1] to image scales
        bboxes_scaled = rescale_bboxes(boxes[0, keep.cpu()], im.size)
        ############ for visualizations


        if keep.nonzero().shape[0] <= 1:
            print("no segmentation")

        # use lists to store the outputs via up-values
        vis_shape, target_shape = [], []

        hooks = [
            self.model.transformer.register_forward_hook(
                lambda self, input, output: vis_shape.append(output[1])
            ),
            self.model.backbone[-2].register_forward_hook(
                lambda self, input, output: target_shape.append(output)
            )
        ]

        self.model(samples)

        for hook in hooks:
            hook.remove()

        h, w = vis_shape[0].shape[-2:]
        # print("h,w", h, w)
        target_shape = target_shape[0]
        import cv2
        masks = torch.ones(1, 100, h, w).to(outputs["pred_logits"].device) * (-1)
        for idx in keep.nonzero():
            if method == 'ours_with_lrp':
                cam = self.gen.generate_ours(samples, idx, use_lrp=True)
            elif method == 'ours_no_lrp':
                cam = self.gen.generate_ours(samples, idx, use_lrp=False)
            elif method == 'ablation_no_self_in_10':
                cam = self.gen.generate_ours(samples, idx, use_lrp=False, apply_self_in_rule_10=False)
            elif method == 'ablation_no_aggregation':
                cam = self.abl.generate_ours_abl(samples, idx, use_lrp=False, normalize_self_attention=False)
            elif method == 'ours_no_lrp_no_norm':
                cam = self.gen.generate_ours(samples, idx, use_lrp=False, normalize_self_attention=False)
            elif method == 'transformer_att':
                cam = self.gen.generate_transformer_att(samples, idx)
            elif method == 'raw_attn':
                cam = self.gen.generate_raw_attn(samples, idx)
            elif method == 'attn_gradcam':
                cam = self.gen.generate_attn_gradcam(samples, idx)
            elif method == 'rollout':
                cam = self.gen.generate_rollout(samples, idx)
            elif method == 'partial_lrp':
                cam = self.gen.generate_partial_lrp(samples, idx)
            else:
                print("please provide a valid explainability method")
                return

            # Otsu
            cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
            Res_img = cam.reshape(h, w)
            Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
            ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cam = torch.from_numpy(th).to(outputs["pred_logits"].device).type(torch.float32)
            masks[0, idx] = cam

        # import matplotlib.pyplot as plt
        #
        # plt.clf()
        # print("Bboxes scaled:::::", len(bboxes_scaled))
        # if len(bboxes_scaled) > 1:
        #     fig, axs = plt.subplots(ncols=len(bboxes_scaled), nrows=2, figsize=(22, 7))
        #     for idx, ax_i, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), axs.T, bboxes_scaled):
        #         ax = ax_i[0]
        #         cam = self.gen.generate_ours(samples, idx, use_lrp=False).reshape(h,w)
        #         cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
        #         Res_img = cam.reshape(h, w)
        #         Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
        #         import cv2
        #         ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #         cam = torch.from_numpy(Res_img)
        #         cam[cam < ret] = 0
        #         cam[cam >= ret] = 1
        #         ax.imshow(cam)
        #         ax.axis('off')
        #         ax.set_title(f'query id: {idx.item()}')
        #         ax = ax_i[1]
        #         ax.imshow(im)
        #         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                                    fill=False, color='blue', linewidth=3))
        #         ax.axis('off')
        #         ax.set_title(CLASSES[probas[idx].argmax()])
        #
        #     plt.show()
        #     plt.savefig('decoder_visualization/_ours_seg.png')
        visualiztion = masks.cpu().numpy()
        return masks


    # Get the explanations
    def get_panoptic_masks_no_thresholding(self, outputs_single_item, idx):
        cam = self.gen.generate_ours_from_outputs(outputs_single_item, idx, use_lrp=False)
        cam = (cam - cam.min()) / (cam.max() - cam.min()) \
              #* 255
        # * 255 was done to get to image range
        return cam

    def get_panoptic_masks_no_thresholding_batchified(self, outputs, idx, h, mask_generator, targets, tgt_idx, w):
        pred_boxes = [outputs["pred_boxes"][im][ind].float().cpu().unsqueeze(0) for im, ind in zip(idx[0], idx[1])]

        pred_boxes = torch.cat(pred_boxes)
        target_boxes = [targets[im]["boxes"][ind].float().cpu().unsqueeze(0) for im, ind in zip(tgt_idx[0], tgt_idx[1])]
        target_boxes = torch.cat(target_boxes)
        target_labels = [targets[im]["labels"][ind].float().cpu().unsqueeze(0) for im, ind in
                         zip(tgt_idx[0], tgt_idx[1])]
        target_labels = torch.cat(target_labels)

        # target_labels_vis = [targets[im]["labels_vis"][ind].float().cpu().unsqueeze(0) for im, ind in
        #                  zip(tgt_idx[0], tgt_idx[1])]
        # target_labels_vis = torch.cat(target_labels_vis)

        pred_probs = [outputs["pred_logits"][im][ind].float().cpu().unsqueeze(0) for im, ind in zip(idx[0], idx[1])]
        pred_probs = torch.cat(pred_probs)

        mask_generator.set_boxes(pred_boxes)
        mask_generator.set_tar_boxes(target_boxes)
        mask_generator.set_probs(pred_probs)
        mask_generator.set_labels(target_labels)
        mask_generator.set_src_idx(idx)
        mask_generator.set_query_ids(idx[1])

        cam = self.gen.generate_ours_from_outputs_batchified(outputs, idx, h, mask_generator, targets, tgt_idx, w, use_lrp=False)
        # normalize !each mask! by its min and max


        # print("DONE WITH REL MAPS!")
        # cam = normalize_rel_maps(cam)
        # cam = cam.unsqueeze(1).unsqueeze(1)
        # cam = (cam - cam.min()) / (cam.max() - cam.min()) \
              #* 255
        # * 255 was done to get to image range

        return cam

    def set_relevance(self,relevance):
        relevance = relevance.detach()
        if(self.relevance is None):
            self.relevance = relevance
        else:
            self.relevance = torch.cat([self.relevance, relevance])

    def get_relevance(self):
        return self.relevance

    def set_orig_rel(self,relevance):
        relevance = relevance.detach()
        if(self.orig_rel is None):
            self.orig_rel = relevance
        else:
            self.orig_rel = torch.cat([self.orig_rel, relevance])

    def get_orig_rel(self):
        return self.orig_rel
    def set_labels(self,labels):
        self.tar_labels = labels.detach()

    def get_labels(self):
        return self.tar_labels

    def set_src_idx(self,labels):
        self.src_idx = labels

    def get_src_idx(self):
        return self.src_idx


    def set_boxes(self, pred_boxes):
        self.pred_boxes = pred_boxes.detach()

    def get_boxes(self):
        return self.pred_boxes

    def set_tar_boxes(self, boxes):
        self.tar_boxes = boxes.detach()

    def get_tar_boxes(self):
        return self.tar_boxes

    def set_targets(self, target_masks):
        target_masks = target_masks.unsqueeze(0).unsqueeze(0).detach()
        if (self.target_masks is None):
            self.target_masks = target_masks
        else:
            self.target_masks = torch.cat([self.target_masks, target_masks])

    def get_targets(self):
        return self.target_masks

    def set_probs(self, probs):
        self.probs = probs.detach()

    def get_probs(self):
        return self.probs

    def set_query_ids(self, query_ids):
        self.query_ids = query_ids.detach()

    def get_query_ids(self):
        return self.query_ids

    def get_panoptic(self, samples, targets, method):
        outputs = self.model(samples)
        masks = self.get_panoptic_masks(outputs,samples, targets, method)
        outputs['pred_masks'] = masks
        return outputs


