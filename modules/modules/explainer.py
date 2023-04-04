import torch
import numpy as np
import cv2

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    img = img.squeeze(0).permute(1,2,0).cpu().numpy()
    mask = mask.squeeze(0).squeeze(0)
    img = (img - img.min()) / (img.max() - img.min())

    heatmap = cv2.applyColorMap(np.uint8(255 * mask.detach().cpu().numpy()), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    # cv2.imshow('image', cam)
    vis = np.uint8(255 * cam)
    # vis =  cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    vis = np.float32(vis)
    return vis


def get_image_with_relevance(image, relevance):

    image = image.squeeze(0)
    relevance = relevance.squeeze(0)
    image = image.permute(1, 2, 0)
    relevance = relevance.permute(1, 2, 0)
    image = (image - image.min()) / (image.max() - image.min())
    image = 255 * image
    vis = image * relevance
    return vis.data.cpu().numpy()