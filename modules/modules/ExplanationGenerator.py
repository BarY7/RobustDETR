import numpy as np
import torch
from torch.nn.functional import softmax

from models.rel_comp import compute_rel_loss_from_map
from util.timer_utils import catchtime


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    eye = torch.eye(num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    matrices_aug = all_layer_matrices
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].matmul(joint_attention)
    return joint_attention

# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam

def avg_batch_heads(cam, grad):
    og_size = cam.shape
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.reshape(*og_size)
    # grad = grad.reshape(*og_size)
    cam = cam.clamp(min=0).mean(dim=2) #should be num_masks x blocks x i x i
    return cam

# rules 6 + 7 from paper
def apply_self_attention_rules(R_ss, R_sq, cam_ss):
    R_sq_addition = torch.bmm(cam_ss, R_sq)
    R_ss_addition = torch.bmm(cam_ss, R_ss)
    return R_ss_addition, R_sq_addition

# rule 10 from paper

# def apply_mm_attention_rules(R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
#     R_ss_normalized = R_ss
#     R_qq_normalized = R_qq
#     if apply_normalization:
#         R_ss_normalized = handle_residual(R_ss)
#         R_qq_normalized = handle_residual(R_qq)
#     #changed from t() TODO verify good
#     R_sq_addition = torch.matmul(torch.transpose(R_ss_normalized,1,2), torch.matmul(cam_sq, R_qq_normalized))
#     if not apply_self_in_rule_10:
#         R_sq_addition = cam_sq
#     R_sq_addition[torch.isnan(R_sq_addition)] = 0
#     return R_sq_additionl.

# normalization- eq. 8+9
# def handle_residual(orig_self_attention):
#     batch_size = orig_self_attention.shape[0]
#     self_attention = orig_self_attention.clone()
#     diag_idx = range(self_attention.shape[-1])
#     self_attention -= torch.eye(self_attention.shape[-1]).to(self_attention.device)
#     assert self_attention[ diag_idx, diag_idx].min() >= 0
#     self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
#     self_attention += torch.cat(batch_size * [torch.eye(self_attention.shape[-1]).unsqueeze(0)]).to(self_attention.device)
#     return self_attention

def handle_residual(orig_self_attention):
    num_masks = orig_self_attention.shape[0]
    self_attention = orig_self_attention.clone()
    diag_idx = range(self_attention.shape[-1])
    self_attention -= torch.eye(self_attention.shape[-1]).repeat(num_masks,1,1).to(self_attention.device)
    assert self_attention[:, diag_idx, diag_idx].min() >= 0
    # PROBLEM - in this line, self.attention.sum contains a row with 0
    # Can make this happen by removing _pre_load_state_dict in transformer and usnig reg model with image 7281
    # x / 0 = NAN which propagates and destroys gradients!
    self_attention = self_attention / self_attention.sum(dim=-1, keepdim=True)
    self_attention = self_attention + torch.eye(self_attention.shape[-1]).repeat(num_masks,1,1).to(self_attention.device)
    return self_attention

class Generator:
    def __init__(self, model):
        self.model = model
        # self.model.eval()
        self.use_lrp = False
        self.normalize_self_attention = True
        self.apply_self_in_rule_10 = True
        self.nan_happened = False


    def set_nan_happpened(self):
        self.nan_happened = True

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

    def apply_mm_attention_rules(self,R_ss, R_qq, cam_sq, apply_normalization=True, apply_self_in_rule_10=True):
        # print(f"R_ss {R_ss}")
        # print(f"R_qq {R_qq}")
        # print(f"cam_sq {cam_sq}")
        R_ss_normalized = R_ss
        R_qq_normalized = R_qq
        if apply_normalization:
            R_ss_normalized = handle_residual(R_ss)
            R_qq_normalized = handle_residual(R_qq)
        R_sq_addition = torch.bmm(torch.transpose(R_ss_normalized, 1, 2), torch.bmm(cam_sq, R_qq_normalized))
        if not apply_self_in_rule_10:
            R_sq_addition = cam_sq
        if torch.isnan(R_sq_addition).any():
            print("GOT NAN! skipping iteration")
            self.set_nan_happpened()
            # print("normalized vals")
            # print(f"R_ss_norm {R_ss_normalized}")
            # print(f"R_qq_norm {R_qq_normalized}")
            # print ("saving transformer module as old_module.pth")
            # raise BaseException("NAN error, saving current model")
        R_sq_addition[torch.isnan(R_sq_addition)] = 0
        return R_sq_addition



    def generate_transformer_att(self, img, target_index, index=None):
        outputs = self.model(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(one_hot_vector, **kwargs)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        # initialize relevancy matrices
        image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
        queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]

        # image self attention matrix
        self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
        # queries self attention matrix
        self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
        # impact of image boxes on queries
        self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)

        # R_q_i generated from last layer
        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn_cam().detach()
        grad_q_i = decoder_last.multihead_attn.get_attn_gradients().detach()
        cam_q_i = avg_heads(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated


    def get_batched_cams_grads(self, blocks, one_hots, all_blocks_cam_list, num_masks):
        all_blocks_cam = torch.stack(all_blocks_cam_list) \
            .unsqueeze(0).expand(num_masks, -1, -1, -1, -1)  # num_masks x num_blocks x num_head x i x i

        all_grads = [torch.autograd.grad(one_hot, all_blocks_cam_list, retain_graph=True)
                     for i, one_hot in enumerate(one_hots)]
        all_grads_flat = [ten for tensor_tuple in all_grads for ten in tensor_tuple]
        all_grads = torch.stack(all_grads_flat)
        all_grads = all_grads.reshape(num_masks, len(blocks), -1,
                                      *all_grads.shape[2:])  # num_masks x num_blocks x num_head x i x i

        return all_blocks_cam, all_grads

    def handle_self_attention_image(self, blocks, one_hots, batch_size=1, req_grad = True):
        num_masks = len(one_hots)
        all_blocks_cam_list = [blk.self_attn.get_attn() for blk in blocks]

        all_blocks_cam, all_grads = self.get_batched_cams_grads(blocks, one_hots, all_blocks_cam_list, num_masks)

        if req_grad:
            avged_heads = avg_batch_heads(all_blocks_cam, all_grads)
        else:
            avged_heads = avg_batch_heads(all_blocks_cam.detach(), all_grads.detach())
        for i in range(len(all_blocks_cam_list)):
            cam = avged_heads[: , i]
            self.R_i_i = self.R_i_i + torch.bmm(cam, self.R_i_i)



        # with catchtime('not super at all') as t:
        # for blk in blocks:
        #     if self.use_lrp:
        #         cam = blk.self_attn.get_attn_cam()
        #     else:
        #         cam = blk.self_attn.get_attn()
        #     grad = torch.autograd.grad(one_hot, [cam], retain_graph=True)[0]
        #
        #
        #     # We don't want to average attention of different batch elements
        #     # num_heads = blk.self_attn.num_heads
        #
        #     # avg heads on same batch size
        #     cam = cam.reshape((-1, cam.shape[-2], cam.shape[-1]))
        #     grad = grad.reshape((-1, grad.shape[-2], grad.shape[-1]))
        #     cam = avg_heads(cam, grad)
        #
        #     # cam = torch.cat([avg_heads(cam[i * num_heads : (i+1) * num_heads,:,:], grad[i * num_heads : (i+1) * num_heads,:,:]).unsqueeze(0) for i in range(batch_size)])
        #
        #     self.R_i_i = self.R_i_i + torch.matmul(cam, self.R_i_i)
        #
        #     del cam
        #     del grad
        #
        # print((self.R_i_i == tmp).all())
        # print("xx")


    def handle_co_attn_self_query_blocks(self,one_hot, cam, req_grad=True):
        # if req_grad:
        #     cam = avg_batch_heads(cam, grad)
        # else:
        #     cam = avg_batch_heads(cam.detach(), grad.detach())
        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q = self.R_q_q + R_q_q_add
        self.R_q_i = self.R_q_i + R_q_i_add
        # print(f"tmp q_q {tmp_q_q}")
        # print(f"tmp q_i {tmp_q_i}")



    def handle_co_attn_self_query_depracated(self, block, one_hot, blocks):

        # with catchtime('MY super versiom') as t:
        #     all_blocks_cam = [blk.self_attn.get_attn() for blk in blocks]
        #     all_grads = torch.stack(torch.autograd.grad(one_hot, all_blocks_cam, retain_graph=True))
        #
        # tmp = self.R_i_i
        # for i in range(len(all_blocks_cam)):
        #     cam = avg_heads(all_blocks_cam[i], all_grads[i])
        #     R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        #     self.R_q_q = self.R_q_q + R_q_q_add
        #     self.R_q_i = self.R_q_i + R_q_i_add
        if self.use_lrp:
            cam = block.self_attn.get_attn_cam()
        else:
            cam = block.self_attn.get_attn()

        grad = torch.autograd.grad(one_hot, [cam], retain_graph=True)[0]

        # We don't want to average attention of different batch elements
        num_heads = block.multihead_attn.num_heads

        cam = cam.reshape((-1, cam.shape[-2], cam.shape[-1]))
        grad = grad.reshape((-1, grad.shape[-2], grad.shape[-1]))
        cam = avg_heads(cam, grad)

        R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
        self.R_q_q = self.R_q_q + R_q_q_add
        self.R_q_i = self.R_q_i +  R_q_i_add

        del R_q_i_add
        del R_q_q_add
        del cam
        del grad
        print(f"Print my result R_q_q {self.R_q_q}")
        print(f"Print my result R_q_i {self.R_q_i}")


    def handle_co_attn_query_blocks(self, one_hot, cam, req_grad = True):
        cam_q_i = cam
        # if req_grad:
        #     cam_q_i = avg_batch_heads(cam, grad)
        # else:
        #     cam_q_i = avg_batch_heads(cam.detach(), grad.detach())
        self.R_q_i = self.R_q_i + self.apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i,
                                                           apply_normalization=self.normalize_self_attention,
                                                           apply_self_in_rule_10=self.apply_self_in_rule_10)


    def handle_co_attn_query(self, block,one_hot, batch_size=1):
        if self.use_lrp:
            cam_q_i = block.multihead_attn.get_attn_cam()
        else:
            cam_q_i = block.multihead_attn.get_attn()
        # grad_q_i = block.multihead_attn.get_attn_gradients().detach()
        grad_q_i = torch.autograd.grad(one_hot, [cam_q_i], retain_graph=True)[0]

        num_heads = block.multihead_attn.num_heads

        cam_q_i = cam_q_i.reshape((-1, cam_q_i.shape[-2], cam_q_i.shape[-1]))
        grad_q_i = grad_q_i.reshape((-1, grad_q_i.shape[-2], grad_q_i.shape[-1]))
        cam_q_i = avg_heads(cam_q_i, grad_q_i)

        self.R_q_i = self.R_q_i + self.apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i,
                                               apply_normalization=self.normalize_self_attention,
                                               apply_self_in_rule_10=self.apply_self_in_rule_10)
        del cam_q_i
        del grad_q_i

    # def generate_ours(self, img, target_index, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True):
    #     self.use_lrp = use_lrp
    #     self.normalize_self_attention = normalize_self_attention
    #     self.apply_self_in_rule_10 = apply_self_in_rule_10
    #     outputs = self.model(img)
    #     outputs = outputs['pred_logits']
    #     kwargs = {"alpha": 1,
    #               "target_index": target_index}
    #
    #     if index == None:
    #         index = outputs[0, target_index, :-1].max(1)[1]
    #
    #     kwargs["target_class"] = index
    #
    #     one_hot = torch.zeros_like(outputs).to(outputs.device)
    #     one_hot[0, target_index, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot.requires_grad_(True)
    #     one_hot = torch.sum(one_hot.cuda() * outputs)
    #
    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)
    #
    #     if use_lrp:
    #         self.model.relprop(one_hot_vector, **kwargs)
    #
    #     decoder_blocks = self.model.transformer.decoder.layers
    #     encoder_blocks = self.model.transformer.encoder.layers
    #
    #     # initialize relevancy matrices
    #     image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
    #     queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]
    #
    #     # image self attention matrix
    #     self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
    #     # queries self attention matrix
    #     self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
    #     # impact of image boxes on queries
    #     self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
    #
    #     # image self attention in the encoder
    #     self.handle_self_attention_image(encoder_blocks)
    #
    #     # decoder self attention of queries followd by multi-modal attention
    #     for blk in decoder_blocks:
    #         # decoder self attention
    #         self.handle_co_attn_self_query(blk)
    #
    #         # encoder decoder attention
    #         self.handle_co_attn_query(blk)
    #     aggregated = self.R_q_i.unsqueeze_(0)
    #
    #     aggregated = aggregated[:,target_index, :].unsqueeze_(0).detach()
    #     return aggregated
    #

    #NOTICE: IF WE HAVE outputs.size[0] > 1 THIS BREAKS!
    # def generate_ours_from_outputs(self, outputs, target_index, index=None, use_lrp=True, normalize_self_attention=True, apply_self_in_rule_10=True):
    #     self.use_lrp = use_lrp
    #     self.normalize_self_attention = normalize_self_attention
    #     self.apply_self_in_rule_10 = apply_self_in_rule_10
    #
    #     outputs = outputs['pred_logits']
    #     kwargs = {"alpha": 1,
    #               "target_index": target_index}
    #
    #     if index == None:
    #         index = outputs[0, target_index, :-1].max(1)[1]
    #
    #     kwargs["target_class"] = index
    #
    #     one_hot = torch.zeros_like(outputs).to(outputs.device)
    #     one_hot[0, target_index, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot.requires_grad_(True)
    #     one_hot = torch.sum(one_hot.cuda() * outputs)
    #
    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)
    #
    #     if use_lrp:
    #         self.model.relprop(one_hot_vector, **kwargs)
    #
    #     decoder_blocks = self.model.transformer.decoder.layers
    #     encoder_blocks = self.model.transformer.encoder.layers
    #
    #     # initialize relevancy matrices
    #     image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
    #     queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]
    #
    #     # image self attention matrix
    #     self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
    #     # queries self attention matrix
    #     self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
    #     # impact of image boxes on queries
    #     self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
    #
    #     # image self attention in the encoder
    #     self.handle_self_attention_image(encoder_blocks)
    #
    #     # decoder self attention of queries followd by multi-modal attention
    #     for blk in decoder_blocks:
    #         # decoder self attention
    #         self.handle_co_attn_self_query(blk)
    #
    #         # encoder decoder attention
    #         self.handle_co_attn_query(blk)
    #     aggregated = self.R_q_i.unsqueeze_(0)
    #
    #     aggregated = aggregated[:,target_index, :].unsqueeze_(0)
    #     # requires_grad = False
    #     return aggregated

    def norm_rel_maps(self, cam):

        tmp = cam.view(cam.size(1), -1)
        min = tmp.min(1, keepdim=True)[0]
        max = tmp.max(1, keepdim=True)[0]
        tmp = (tmp - min) / (max - min)
        cam = tmp.view(cam.shape)
        return cam


    # I removed relprop

    def generate_ours_from_outputs_batchified(self, outputs, batch_target_idx, h, mask_generator, targets, tgt_idx, w, index=None, use_lrp=False, normalize_self_attention=True, apply_self_in_rule_10=True):
        self.use_lrp = use_lrp
        self.normalize_self_attention = normalize_self_attention
        self.apply_self_in_rule_10 = apply_self_in_rule_10

        outputs_logits = outputs['pred_logits']
        batch_size = outputs_logits.shape[0]

        device = outputs_logits.device
        # kwargs = {"alpha": 1,
        #           "target_index": target_index}
        # TODO verify index
        # if index == None:
        #     index = outputs[batch_target_idx[0], batch_target_idx[1], :-1].argmax(1)
            # index = outputs[batch_target_idx[0][0], batch_target_idx[1][0], :-1].argmax(0)
        #
        # kwargs["target_class"] = index

        agg_list = []

        # l = 0

        for img_idx, mask_idx, tgt_img_idx, tgt_mask_idx in zip(batch_target_idx[0],batch_target_idx[1], tgt_idx[0], tgt_idx[1]):
            # print(torch.cuda.memory_summary())
            # index = outputs[batch_target_idx[0], batch_target_idx[1], :-1].argmax(1)
            with catchtime(f'Comp loss {mask_idx}'):
                mask_idx_l = [mask_idx]

                cam = self.compute_normalized_rel_map_iter(batch_size, img_idx, mask_idx_l, outputs_logits)

                l = compute_rel_loss_from_map(outputs_logits, batch_target_idx, h, mask_generator, cam, targets, tgt_idx, w,
                                              tgt_img_idx, tgt_mask_idx,)
                l = l * mask_generator.get_weight_coef()
                # print(torch.cuda.memory_summary())
                if mask_generator.is_train_mode() and not mask_generator.should_skip_backward():
                    # print(f"Printing grads BE4 backwards of img {img_idx} mask {mask_idx} img_id {targets[img_idx]['image_id']}")
                    # print(self.model.transformer.decoder.get_parameter('layers.0.multihead_attn.k_proj.weight').grad)
                    l.backward(retain_graph=True)
                    # print(f"Printing grads AFTER backwards of img {img_idx} mask {mask_idx} img_id {targets[img_idx]['image_id']}")
                    # print(self.model.transformer.decoder.get_parameter('layers.0.multihead_attn.k_proj.weight').grad)
                    # print(torch.isnan(
                    #     self.model.transformer.decoder.get_parameter('layers.0.multihead_attn.k_proj.weight').grad).any())
                    isnan_list = []
                    nograd_list = []
                    # for k, v in self.model.transformer.decoder.named_parameters():
                    #     if self.model.transformer.decoder.get_parameter(k).grad is None:
                    #         nograd_list.append(k)
                    #     else:
                    #         isnan_list.append(torch.isnan(
                    #             self.model.transformer.decoder.get_parameter(k).grad).any())
                    # print(isnan_list)

                    # if(self.model.transformer.decoder.get_parameter('layers.0.multihead_attn.k_proj.weight').grad )
                    # print(torch.cuda.memory_summary())
                elif mask_generator.should_skip_backward():
                    print("ERR - SKIPPED BACKWARDS FOR ")
                mask_generator.reset_nan_happened()
                agg_list.append(torch.tensor(l.detach().item()))

            del l
            del self.R_q_i
            del self.R_i_i
            del self.R_q_q

            # loss_batch_count += 1
            # if loss_batch_count == 3 :
            #     loss_batch_count = 0
            #     l = l * mask_generator.get_weight_coef()
            #     l.backward(retain_graph=True)
            #     agg_list.append(torch.tensor(l.detach().item()))
            #     del l
            #     l = 0

        # l.backward(retain_graph=True)
        #NOW EVERYTHING CAN BE SAFELY DELETED.
        # if l != 0:
        #     loss_batch_count = 0
        #     l.backward(retain_graph=True)
        #     agg_list.append(torch.tensor(l.detach().item()))
        #     l = 0
        # print(l)


        return torch.tensor(agg_list).to(device).sum()

    def compute_normalized_rel_map_iter(self, batch_size, img_idx, mask_idx_batched, outputs_logits, index = None, req_grad = True):
        num_masks = len(mask_idx_batched)
        with catchtime(f'ALL FUNC TIME , bs={num_masks}'):
            device = outputs_logits.device
            if(index is None):
                # index = outputs_logits[img_idx, mask_idx, :-1].argmax(0)
                index = outputs_logits[img_idx, mask_idx_batched, :].argmax(1)
            one_hots = []
            with catchtime(f'One hots, bs={num_masks}'):
                for i in range(len(mask_idx_batched)):
                    one_hot = torch.zeros_like(outputs_logits).to(outputs_logits.device)
                    one_hot[img_idx, mask_idx_batched[i], index[i]] = 1
                    # one_hot[batch_target_idx[0][0], batch_target_idx[1][0], index] = 1
                    # one_hot_vector = one_hot
                    one_hot.requires_grad_(True)
                    one_hot = torch.sum(one_hot.cuda() * outputs_logits)
                    one_hots.append(one_hot)
            # self.model.zero_grad()
            # one_hot.backward(retain_graph=True)
            # if use_lrp:
            # self.model.relprop(one_hot_vector, **kwargs)
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_no_ddp = self.model.module
            else:
                model_no_ddp = self.model

            decoder_blocks = model_no_ddp.transformer.decoder.layers
            encoder_blocks = model_no_ddp.transformer.encoder.layers
            # initialize relevancy matrices
            image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
            queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]
            # device = encoder_blocks[0].self_attn.get_attn().device
            # image self attention matrix
            self.R_i_i = torch.eye(image_bboxes, image_bboxes).repeat(num_masks, 1, 1)\
                .to(device)
            # queries self attention matrix
            self.R_q_q = torch.eye(queries_num, queries_num).repeat(num_masks, 1, 1)\
                .to(device)
            # impact of image boxes on queries
            self.R_q_i = torch.zeros(queries_num, image_bboxes).repeat(num_masks, 1, 1)\
                .to(device)
            # image self attention in the encoder

            if not req_grad:
                self.R_i_i = self.R_i_i.detach()
                self.R_q_q = self.R_q_q.detach()
                self.R_q_i = self.R_q_i.detach()

            num_dec_blocks = len(decoder_blocks)
            with catchtime(f'Self attn, bs={num_masks}'):
                self.handle_self_attention_image(encoder_blocks, one_hots, batch_size, req_grad)

            self_blocks_cam = [blk.self_attn.get_attn() for blk in decoder_blocks]
            multihead_blocks_cam = [blk.multihead_attn.get_attn() for blk in decoder_blocks]

            # self_dec_grads = torch.autograd.grad(one_hot, self_blocks_cam, retain_graph=True)
            # multihead_dec_grads = torch.autograd.grad(one_hot, multihead_blocks_cam, retain_graph=True)
            with catchtime(f'Get grads, bs={num_masks}'):
                self_batch_cam, self_dec_grads = self.get_batched_cams_grads(decoder_blocks, one_hots, self_blocks_cam,
                                                                             num_masks)
                multihead_batch_cam, multihead_dec_grads = self.get_batched_cams_grads(decoder_blocks, one_hots, multihead_blocks_cam,
                                                                         num_masks)
            if req_grad:
                avged_heads_mult = avg_batch_heads(multihead_batch_cam, multihead_dec_grads)
                avged_heads_self = avg_batch_heads(self_batch_cam, self_dec_grads)
            else:
                avged_heads_mult = avg_batch_heads(multihead_batch_cam.detach(), multihead_dec_grads.detach())
                avged_heads_self = avg_batch_heads(self_batch_cam.detach(), self_dec_grads.detach())
            with catchtime(f'Compute Rs, bs={num_masks} , index = {i}'):
                for i in range(num_dec_blocks):
                    self.handle_co_attn_self_query_blocks(one_hot, avged_heads_self[:, i], req_grad)
                    self.handle_co_attn_query_blocks(one_hot, avged_heads_mult[:, i], req_grad)

            aggregated = self.R_q_i
            aggregated = aggregated[torch.arange(num_masks) , mask_idx_batched].unsqueeze(0)
            # agg_list.append(aggregated)
            # requires_grad = False
            # del self.R_q_q
            # del self.R_q_i
            # del self.R_i_i
            # del one_hot
            # del self.R_i_i
            # del self.R_q_q
            # del self.R_q_i
            with catchtime(f'Norm, bs={num_masks}'):
                cam = self.norm_rel_maps(aggregated)
        return cam

    def generate_partial_lrp(self, img, target_index, index=None):
        outputs = self.model(img)
        kwargs = {"alpha": 1,
                  "target_index": target_index}

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        kwargs["target_class"] = index

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot_vector = one_hot.clone().detach()

        self.model.relprop(one_hot_vector, **kwargs)

        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn_cam().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = cam_q_i

        # normalize to get non-negative cams
        self.R_q_i = (self.R_q_i - self.R_q_i.min()) / (self.R_q_i.max() - self.R_q_i.min())
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def generate_raw_attn(self, img, target_index):
        outputs = self.model(img)

        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = cam_q_i

        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def generate_rollout(self, img, target_index):
        outputs = self.model(img)

        decoder_blocks = self.model.transformer.decoder.layers
        encoder_blocks = self.model.transformer.encoder.layers

        cams_image = []
        cams_queries = []
        # image self attention in the encoder
        for blk in encoder_blocks:
            cam = blk.self_attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_image.append(cam)

        # decoder self attention of queries
        for blk in decoder_blocks:
            # decoder self attention
            cam = blk.self_attn.get_attn().detach()
            cam = cam.mean(dim=0)
            cams_queries.append(cam)

        # rollout for self-attention values
        self.R_i_i = compute_rollout_attention(cams_image)
        self.R_q_q = compute_rollout_attention(cams_queries)

        decoder_last = decoder_blocks[-1]
        cam_q_i = decoder_last.multihead_attn.get_attn().detach()
        cam_q_i = cam_q_i.reshape(-1, cam_q_i.shape[-2], cam_q_i.shape[-1])
        cam_q_i = cam_q_i.mean(dim=0)
        self.R_q_i = torch.matmul(self.R_q_q.t(), torch.matmul(cam_q_i, self.R_i_i))
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated

    def gradcam(self, cam, grad):
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        return cam

    def generate_attn_gradcam(self, img, target_index, index=None):
        outputs = self.model(img)

        if index == None:
            index = outputs['pred_logits'][0, target_index, :-1].max(1)[1]

        one_hot = torch.zeros_like(outputs['pred_logits']).to(outputs['pred_logits'].device)
        one_hot[0, target_index, index] = 1
        one_hot.requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * outputs['pred_logits'])

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)


        # get cross attn cam from last decoder layer
        cam_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn().detach()
        grad_q_i = self.model.transformer.decoder.layers[-1].multihead_attn.get_attn_gradients().detach()
        cam_q_i = self.gradcam(cam_q_i, grad_q_i)
        self.R_q_i = cam_q_i
        aggregated = self.R_q_i.unsqueeze_(0)

        aggregated = aggregated[:, target_index, :].unsqueeze_(0)
        return aggregated




class GeneratorAlbationNoAgg:
    def __init__(self, model):
        self.model = model
        # self.model.eval()

    # def forward(self, input_ids, attention_mask):
    #     return self.model(input_ids, attention_mask)
    #
    # def handle_self_attention_image(self, blocks):
    #     for blk in blocks:
    #         grad = blk.self_attn.get_attn_gradients().detach()
    #         if self.use_lrp:
    #             cam = blk.self_attn.get_attn_cam().detach()
    #         else:
    #             cam = blk.self_attn.get_attn().detach()
    #         cam = avg_heads(cam, grad)
    #         self.R_i_i = torch.matmul(cam, self.R_i_i)
    #
    # def handle_co_attn_self_query(self, block):
    #     with catchtime('MY super versiom') as t:
    #         all_blocks_cam = [blk.self_attn.get_attn() for blk in blocks]
    #         all_grads = torch.stack(torch.autograd.grad(one_hot, all_blocks_cam, retain_graph=True))
    #
    #     tmp = self.R_i_i
    #     for i in range(len(all_blocks_cam)):
    #         cam = avg_heads(all_blocks_cam[i], all_grads[i])
    #         tmp = tmp + torch.matmul(cam, tmp)
    #
    #
    #     grad = block.self_attn.get_attn_gradients().detach()
    #     if self.use_lrp:
    #         cam = block.self_attn.get_attn_cam().detach()
    #     else:
    #         cam = block.self_attn.get_attn().detach()
    #     cam = avg_heads(cam, grad)
    #     R_q_q_add, R_q_i_add = apply_self_attention_rules(self.R_q_q, self.R_q_i, cam)
    #     self.R_q_q = R_q_q_add
    #     self.R_q_i = R_q_i_add
    #
    # def handle_co_attn_query(self, block):
    #     if self.use_lrp:
    #         cam_q_i = block.multihead_attn.get_attn_cam().detach()
    #     else:
    #         cam_q_i = block.multihead_attn.get_attn().detach()
    #     grad_q_i = block.multihead_attn.get_attn_gradients().detach()
    #     cam_q_i = avg_heads(cam_q_i, grad_q_i)
    #     self.R_q_i = apply_mm_attention_rules(self.R_q_q, self.R_i_i, cam_q_i,
    #                                            apply_normalization=self.normalize_self_attention,
    #                                            apply_self_in_rule_10=self.apply_self_in_rule_10)
    #
    # def generate_ours_abl(self, img, target_index, index=None, use_lrp=False, normalize_self_attention=False, apply_self_in_rule_10=True):
    #     self.use_lrp = use_lrp
    #     self.normalize_self_attention = normalize_self_attention
    #     self.apply_self_in_rule_10 = apply_self_in_rule_10
    #     outputs = self.model(img)
    #     outputs = outputs['pred_logits']
    #     kwargs = {"alpha": 1,
    #               "target_index": target_index}
    #
    #     if index == None:
    #         index = outputs[0, target_index, :-1].max(1)[1]
    #
    #     kwargs["target_class"] = index
    #
    #     one_hot = torch.zeros_like(outputs).to(outputs.device)
    #     one_hot[0, target_index, index] = 1
    #     one_hot_vector = one_hot
    #     one_hot.requires_grad_(True)
    #     one_hot = torch.sum(one_hot.cuda() * outputs)
    #
    #     self.model.zero_grad()
    #     one_hot.backward(retain_graph=True)
    #
    #     if use_lrp:
    #         self.model.relprop(one_hot_vector, **kwargs)
    #
    #     decoder_blocks = self.model.transformer.decoder.layers
    #     encoder_blocks = self.model.transformer.encoder.layers
    #
    #     # initialize relevancy matrices
    #     image_bboxes = encoder_blocks[0].self_attn.get_attn().shape[-1]
    #     queries_num = decoder_blocks[0].self_attn.get_attn().shape[-1]
    #
    #     # image self attention matrix
    #     self.R_i_i = torch.eye(image_bboxes, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
    #     # queries self attention matrix
    #     self.R_q_q = torch.eye(queries_num, queries_num).to(encoder_blocks[0].self_attn.get_attn().device)
    #     # impact of image boxes on queries
    #     self.R_q_i = torch.zeros(queries_num, image_bboxes).to(encoder_blocks[0].self_attn.get_attn().device)
    #
    #     # image self attention in the encoder
    #     self.handle_self_attention_image(encoder_blocks)
    #
    #     # decoder self attention of queries followd by multi-modal attention
    #     for blk in decoder_blocks:
    #         # decoder self attention
    #         self.handle_co_attn_self_query(blk)
    #
    #         # encoder decoder attention
    #         self.handle_co_attn_query(blk)
    #     aggregated = self.R_q_i.unsqueeze_(0)
    #
    #     aggregated = aggregated[:,target_index, :].unsqueeze_(0).detach()
    #     return aggregated
