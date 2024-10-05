import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import cv2
from tqdm import tqdm
 
# For FastSAM
from fastsam import FastSAM
import timm 



class ZeroWasteDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform
 
        # Load all image file names
        self.image_files = sorted(os.listdir(self.data_dir))
 
        # Load annotations
        with open(os.path.join(self.label_dir, 'labels.json'), 'r') as f:
            self.labels = json.load(f)
 
        # Build a mapping from image_id to annotations
        self.image_id_to_annotations = {}
        for ann in self.labels['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)
 
        # Build a mapping from image_id to file_name
        self.image_id_to_file_name = {img['id']: img['file_name'] for img in self.labels['images']}
 
        # Build a mapping from category_id to category_name
        self.category_id_to_name = {cat['id']: cat['name'] for cat in self.labels['categories']}
 
    def __len__(self):
        return len(self.labels['images'])
 
    def __getitem__(self, idx):
        # Get image ID and file name
        image_info = self.labels['images'][idx]
        image_id = image_info['id']
        file_name = image_info['file_name']
 
        # Load image
        image_path = os.path.join(self.data_dir, file_name)
        image = Image.open(image_path).convert('RGB')
 
        # Load annotations for this image
        anns = self.image_id_to_annotations.get(image_id, [])
 
        # Apply transform if any
        if self.transform:
            image = self.transform(image)
        else:
            # convert image to numpy array
            image = np.array(image)
            #convert numpy array to tensor
            image = torch.from_numpy(image)
            
        
        return image, anns, image_id
 
# Define your transforms (include Resize and ToTensor)
train_transform = transforms.Compose([
    # transforms.Resize((640, 480)),  # Resize images to a fixed size
    # transforms.ToTensor(),
    # Add other transformations if needed
])
 
train_dataset = ZeroWasteDataset(
    data_dir='dataset/train/data/',
    label_dir='dataset/train/'
)
 
def custom_collate_fn(batch):
    images, anns, image_ids = zip(*batch)
 
    # Stack images (they are tensors of the same size)
    images = torch.stack(images, dim=0)
 
    # anns is a tuple of lists (variable lengths), so we keep them as a list
    anns = list(anns)
 
    # image_ids can be converted to a list or tensor
    image_ids = torch.tensor(image_ids)
 
    return images, anns, image_ids
 
train_loader = DataLoader(
    train_dataset,
    batch_size=12,
    shuffle=False,
    collate_fn=custom_collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fast_sam_model = FastSAM('weights/FastSAM.pt')
fast_sam_model.to(device)

def segment_image_with_fastsam(image):
    # Convert PIL image to the format expected by FastSAM
    mask_result = fast_sam_model(
        image,
        device=device,
        retina_masks=True,
        imgsz=[1920, 1080],
        conf=0.3,
        iou=0.9
    )    
    return mask_result

import pycocotools.mask as maskUtils

 
def match_masks_to_annotations(fast_sam_masks, ground_truth_anns, image_size, iou_threshold=0.5):

    # Decode ground truth masks

    gt_masks = []

    gt_labels = []

    for ann in ground_truth_anns:

        rle = maskUtils.frPyObjects(ann['segmentation'], image_size[0], image_size[1])

        gt_mask = maskUtils.decode(rle)

        gt_masks.append(gt_mask)

        gt_labels.append(ann['category_id'])

    # FastSAM masks

    fs_masks = fast_sam_masks.masks.data.cpu().numpy()  # Shape: (N, H, W)

    matched_labels = []

    for fs_mask in fs_masks:

        max_iou = 0

        matched_label = None

        for gt_mask, gt_label in zip(gt_masks, gt_labels):

            intersection = np.logical_and(fs_mask, gt_mask)

            union = np.logical_or(fs_mask, gt_mask)

            iou = np.sum(intersection) / np.sum(union)

            if iou > max_iou and iou > iou_threshold:

                max_iou = iou

                matched_label = gt_label

        matched_labels.append(matched_label)

    return matched_labels

# dinov2_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device)
# dinov2_model.eval()

 
# def extract_features_with_dinov2(object_image):

#     transform = transforms.Compose([

#         transforms.Resize((224, 224)),

#         transforms.ToTensor(),

#         transforms.Normalize(

#             mean=[0.485, 0.456, 0.406],  # Adjust if necessary

#             std=[0.229, 0.224, 0.225]

#         )

#     ])

#     input_tensor = transform(object_image).unsqueeze(0).to(device)

#     with torch.no_grad():

#         features = dinov2_model.forward_features(input_tensor)

#     # Flatten features if necessary

#     features = features.view(features.size(0), -1)

#     return features.cpu()


def extract_object_from_mask(image, mask):
    # Convert mask to PIL image
    mask_pil = Image.fromarray((mask * 255).astype('uint8'), mode='L')
    # Apply mask to the image
    object_image = Image.composite(image, Image.new('RGB', image.size), mask_pil)
    return object_image


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = self.fc(x)
        return x 
    
    
features_list = []
labels_list = []


from pycocotools import mask
from mmdet.core.visualization.image import imshow_det_bboxes
import pycocotools.mask as maskUtils
import mmcv


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
        post_processed_result['segmentation']['size'] = rle['size']
        post_processed_result['area'] = mask.area(rle)

        post_processed_result['predicted_iou'] = 0  # placeholder
        post_processed_result['point_coords'] = [0, 0]  # placeholder
        post_processed_result['stability_score'] = 0  # placeholder

        post_processed_result['crop_box'] = [0, 0, result.orig_shape[1], result.orig_shape[0]]
        post_processed_results.append(post_processed_result)

    return post_processed_results

 
for image, anns, image_id in tqdm(train_loader):
    # Segment image
    # convert image to PIL image
    np_image = image.numpy().astype('uint8')
    image = [Image.fromarray(image[i].numpy().astype('uint8'), mode='RGB') for i in range(image.shape[0])]
    mask_result = segment_image_with_fastsam(image)
    if mask_result is None:
        print(f"None result for image {image_id}")
        continue

    for i, res in enumerate(mask_result):
        image = mmcv.imread(np_image[i])
        anns = {'annotations': postprocess_fastSAM(res)}        
        bitmasks = []
        for ann in anns['annotations']:
            bitmasks.append(maskUtils.decode(ann['segmentation']))
            
        imshow_det_bboxes(
                image,
                bboxes=None,
                labels=np.arange(len(bitmasks)),
                segms=np.stack(bitmasks),
                font_size=25,
                show=False,
                out_file=f"output_{image_id[i]}.png"
        )
        
    # Get masks
    # fs_masks = mask_result.masks.data.cpu().numpy()  # (N, H, W)
    # # Match masks to annotations
    # matched_labels = match_masks_to_annotations(mask_result, anns, image.size)
    # Extract features for each mask
    # for idx, fs_mask in enumerate(fs_masks):
    #     label = matched_labels[idx]
    #     if label is not None:
    #         # Extract object using the mask
    #         object_image = extract_object_from_mask(image, fs_mask)
    #         # Extract features
    #         features = extract_features_with_dinov2(object_image)
    #         features_list.append(features)
    #         labels_list.append(label)