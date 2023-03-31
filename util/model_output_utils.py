import torch
import cv2
import numpy as np

from typing import Dict


def get_one_output_from_batch(in_dict, i):
    new_in_dict: Dict = {}
    for key in in_dict:
        if torch.is_tensor(in_dict[key]):
            new_in_dict[key] = in_dict[key][i].unsqueeze(0)
        else:
            new_in_dict[key] = in_dict[key]
    return new_in_dict


def normalize_rel_maps(cam):
    tmp = cam.view(cam.size(0), -1)
    min = tmp.min(1, keepdim=True)[0]
    max = tmp.max(1, keepdim=True)[0]
    tmp = (tmp - min) / (max - min)
    cam = tmp.view(cam.shape)
    return cam

def otsu_thresh(cam):
    # Otsu
    dev = cam.device
    cam = (cam - cam.min()) / (cam.max() - cam.min()) * 255
    Res_img = cam
    Res_img = Res_img.data.cpu().numpy().astype(np.uint8)
    ret, th = cv2.threshold(Res_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cam = torch.from_numpy(th).to(dev).type(torch.float32)
    return cam
