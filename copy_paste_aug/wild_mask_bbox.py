
import os
import pandas as pd
import json
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

def get_bbox(bbox_df,image_id):
    try:
        bbox, conf = bbox_df.loc[image_id]['detections'], bbox_df.loc[image_id]['max_detection_conf']
        bbox = [b['bbox'] for b in bbox]
        if conf>0.5 and len(bbox)>0:
            return bbox, conf
        else:
            return [], 0.0
    except:
        return [], 0.0


def get_mask(mask_dir,image_id, resize_wh=None, pad_wh=None):
    try:
        mask = Image.open(str(mask_dir)+'//'+str(image_id)+'.png')
        if resize_wh is not None:
            mask = mask.resize(resize_wh[:2])
        mask = F.to_tensor(mask)
        if pad_wh is not None:
            w, h = mask.shape
            gap_w, gap_h = pad_wh[0] - w, pad_wh[1] - h
            F.pad(mask, (gap_w/2, gap_h/2, gap_w/2, gap_h/2), fill=0)
        mask = mask.squeeze().numpy() > 0
        return Image.fromarray(mask)
    except:
        return None


def xywh_to_xyxy(x, y, w, h, xy_center=False):
    """convert [x,y,w,h] -> [x_1, y_1, x_2, y_2]"""
    if xy_center:
        # (x,y) gives the center of the box
        x1, y1 = x-w/2, y-h/2
        x2, y2 = x+w/2, y+h/2
    else:
        # (x,y) gives the upper left corner of the box
        x1, y1 = x, y
        x2, y2 = x+w, y+h
    return int(x1), int(y1), int(x2), int(y2)

def create_mask_from_bboxes(bboxes, mask_wh: Image.Image):
    """Create an Image mask from a set of bboxes"""
    img_w, img_h = mask_wh
    mask_im = Image.new("L", mask_wh, 0) # make a mask & fill with black
    for bbox in bboxes:
        x,y,w,h = bbox
        draw = ImageDraw.Draw(mask_im) # make canvas
        xyxy = xywh_to_xyxy(
            x * img_w,  # fraction -> pixels
            y * img_h,
            w * img_w,
            h * img_h,
        )
        draw.rectangle(xyxy, fill=255) # draw a rectangle of white where bbox is
    mask_im = mask_im.filter(ImageFilter.GaussianBlur(5))
    return mask_im


def generate_final_mask(mask_dir,bbox_df,image_name,mask_img_ids,bbox_img_ids):
    mask=None
    mask_value=np.zeros((448,448))
    # if the segmentation mask exist, then use masks
    if image_name in mask_img_ids:
        mask= get_mask(mask_dir,image_name, resize_wh=(448,448))
        mask_value=np.asarray(mask)
    # if the mask does not exist but bounding box exist, use the mask induced from the bounding box
    else:
        if image_name in bbox_img_ids:
            bbox,_=get_bbox(bbox_df,image_name)
            if bbox!=[]:
                mask = create_mask_from_bboxes(bbox, (448,448))
                mask_value=np.asarray(mask)
    # both do not exist, output mask=None
    return mask,mask_value
