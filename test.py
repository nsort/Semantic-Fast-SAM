import os
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv

from fastsam import FastSAM
from pycocotools import mask

import numpy as np

import cv2
import matplotlib.pyplot as plt
from mmdet.core.visualization.image import imshow_det_bboxes
import pycocotools.mask as maskUtils



def postprocess_fastSAM(result, threshold=0.5):

    boxes = result.boxes.data  # get bounding boxes
    masks = result.masks.data  # get masks

    # List to hold post-processed results for each object
    post_processed_results = []

    for i in range(masks.shape[0]):  # Iterate over all detected objects
        post_processed_result = {}

        post_processed_result['bbox'] = boxes[i].tolist()  # convert to list for JSON serialization
        post_processed_result['segmentation'] = {}  # The segmentation info is not directly available in FastSAM 

        binary_mask = (masks[i] > threshold).type(torch.uint8)
        rle = mask.encode(np.asfortranarray(binary_mask.cpu().numpy()))  
        post_processed_result['segmentation']['counts'] = rle['counts']
        post_processed_result['area'] = mask.area(rle)

        post_processed_result['predicted_iou'] = 0  # placeholder
        post_processed_result['point_coords'] = [0, 0]  # placeholder
        post_processed_result['stability_score'] = 0  # placeholder

        post_processed_result['crop_box'] = [0, 0, result.orig_shape[1], result.orig_shape[0]]
        post_processed_results.append(post_processed_result)

    return post_processed_results

if __name__ == "__main__":

    model = FastSAM('weights/FastSAM.pt')
    img_path_fast = 'imgs/rubish.jpg'
    input = Image.open(img_path_fast)
    input = input.convert("RGB")
    input.show()
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    mask_result = model(
        input,
        device=device,
        retina_masks=True,
        imgsz=1024,
        conf=0.3,
        iou=0.7    
        )
    bitmasks = []
    

    
    img = mmcv.imread(img_path_fast)
    anns = {'annotations': postprocess_fastSAM(mask_result[0])}
    for ann in anns['annotations']:
            bitmasks.append(maskUtils.decode(ann['segmentation']))
            imshow_det_bboxes(img,
                bboxes=None,
                labels=np.arange(len(bitmasks)),
                segms=np.stack(bitmasks),
                font_size=25,
                show=True,
                out_file='semantic.png')
    # print(anns)
    
   

    
