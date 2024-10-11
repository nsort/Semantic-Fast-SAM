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

# from pycocotools import mask
# from mmdet.core.visualization.image import imshow_det_bboxes
# import pycocotools.mask as maskUtils
# import mmcv



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class ZeroWasteDataset(Dataset):
#     def __init__(self, data_dir, label_dir, transform=None):
#         self.data_dir = data_dir
#         self.label_dir = label_dir
#         self.transform = transform
 
#         # Load all image file names
#         self.image_files = sorted(os.listdir(self.data_dir))
 
#         # Load annotations
#         with open(os.path.join(self.label_dir, 'labels.json'), 'r') as f:
#             self.labels = json.load(f)
 
#         # Build a mapping from image_id to annotations
#         self.image_id_to_annotations = {}
#         for ann in self.labels['annotations']:
#             image_id = ann['image_id']
#             if image_id not in self.image_id_to_annotations:
#                 self.image_id_to_annotations[image_id] = []
#             self.image_id_to_annotations[image_id].append(ann)
 
#         # Build a mapping from image_id to file_name
#         self.image_id_to_file_name = {img['id']: img['file_name'] for img in self.labels['images']}
 
#         # Build a mapping from category_id to category_name
#         self.category_id_to_name = {cat['id']: cat['name'] for cat in self.labels['categories']}
 
#     def __len__(self):
#         return len(self.labels['images'])
 
#     def __getitem__(self, idx):
#         # Get image ID and file name
#         image_info = self.labels['images'][idx]
#         image_id = image_info['id']
#         file_name = image_info['file_name']
 
#         # Load image
#         image_path = os.path.join(self.data_dir, file_name)
#         image = Image.open(image_path).convert('RGB')
 
#         # Load annotations for this image
#         anns = self.image_id_to_annotations.get(image_id, [])
 
#         # Apply transform if any
#         if self.transform:
#             image = self.transform(image)
#         else:
#             # convert image to numpy array
#             image = np.array(image)
#             #convert numpy array to tensor
#             image = torch.from_numpy(image)
            
        
#         return image, anns, image_id
 
# train_dataset = ZeroWasteDataset(
#     data_dir='dataset/train/data/',
#     label_dir='dataset/train/'
# )
 
# def custom_collate_fn(batch):
#     images, anns, image_ids = zip(*batch)
 
#     # Stack images (they are tensors of the same size)
#     images = torch.stack(images, dim=0)
 
#     # anns is a tuple of lists (variable lengths), so we keep them as a list
#     anns = list(anns)
 
#     # image_ids can be converted to a list or tensor
#     image_ids = torch.tensor(image_ids)
 
#     return images, anns, image_ids
 
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=13,
#     shuffle=False,
#     collate_fn=custom_collate_fn
# )

# fast_sam_model = FastSAM('weights/FastSAM.pt')
# fast_sam_model.to(device)

# dinov2_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device)
# dinov2_model.eval()

# def segment_image_with_fastsam(image):
#     # Convert PIL image to the format expected by FastSAM
#     mask_result = fast_sam_model(
#         image,
#         device=device,
#         retina_masks=True,
#         imgsz=[1920, 1080],
#         conf=0.8,
#         iou=0.8
#     )    
#     return mask_result

# import pycocotools.mask as maskUtils

 
# def match_masks_to_annotations(fast_sam_masks, ground_truth_anns, image_size, iou_threshold=0.5):

#     # Decode ground truth masks

#     gt_masks = []

#     gt_labels = []

#     for ann in ground_truth_anns:

#         rle = maskUtils.frPyObjects(ann['segmentation'], image_size[0], image_size[1])

#         gt_mask = maskUtils.decode(rle)

#         gt_masks.append(gt_mask)

#         gt_labels.append(ann['category_id'])

#     # FastSAM masks

#     fs_masks = fast_sam_masks.masks.data.cpu().numpy()  # Shape: (N, H, W)

#     matched_labels = []

#     for fs_mask in fs_masks:

#         max_iou = 0

#         matched_label = None

#         for gt_mask, gt_label in zip(gt_masks, gt_labels):

#             intersection = np.logical_and(fs_mask, gt_mask)

#             union = np.logical_or(fs_mask, gt_mask)

#             iou = np.sum(intersection) / np.sum(union)

#             if iou > max_iou and iou > iou_threshold:

#                 max_iou = iou

#                 matched_label = gt_label

#         matched_labels.append(matched_label)

#     return matched_labels

 
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

#         # features = dinov2_model.forward_features(input_tensor)
#         features = dinov2_model(input_tensor)
        

#     # Flatten features if necessary

#     features = features.view(features.size(0), -1)

#     return features.cpu()


# iou_threshold = 0.5 # Define IoU threshold for matching

# def segment_image_with_fastsam(image):
#     width, height = image.size
#     mask_result = fast_sam_model(
#         image,
#         device=device,
#         retina_masks=True,
#         imgsz=[width, height],  # Use original image size
#         conf=0.6,
#         iou=0.8
#     )
#     return mask_result

# def extract_object_from_mask(image, mask):
#     # Convert mask to PIL image
#     mask_pil = Image.fromarray((mask * 255).astype('uint8'), mode='L')
#     # Apply mask to the image
#     object_image = Image.composite(image, Image.new('RGB', image.size), mask_pil)
#     return object_image


# def postprocess_fastSAM(result, threshold=0.5):

#     boxes = result.boxes.data  # get bounding boxes
#     masks = result.masks.data  # get masks

#     # List to hold post-processed results for each object
#     post_processed_results = []

#     for i in range(masks.shape[0]):  # Iterate over all detected objects
#         post_processed_result = {}

#         post_processed_result['bbox'] = boxes[i].tolist()  # convert to list for JSON serialization
#         post_processed_result['segmentation'] = {}  # The segmentation info is not directly available in FastSAM 

#         binary_mask = (masks[i] > threshold).type(torch.uint8)
#         rle = mask.encode(np.asfortranarray(binary_mask.cpu().numpy()))  
#         post_processed_result['segmentation']['counts'] = rle['counts']
#         post_processed_result['segmentation']['size'] = rle['size']
#         post_processed_result['area'] = mask.area(rle)

#         post_processed_result['predicted_iou'] = 0  # placeholder
#         post_processed_result['point_coords'] = [0, 0]  # placeholder
#         post_processed_result['stability_score'] = 0  # placeholder

#         post_processed_result['crop_box'] = [0, 0, result.orig_shape[1], result.orig_shape[0]]
#         post_processed_results.append(post_processed_result)

#     return post_processed_results



# # Initialize lists to collect features and labels
# features_list = []
# labels_list = []

# counter = 0
# # Main processing loop
# for images_batch, anns_batch, image_ids_batch in tqdm(train_loader):
    
#     if counter>20:
#         break
#     counter += 1
    
#     batch_size = images_batch.shape[0]
#     for idx in range(batch_size):
#         image_tensor = images_batch[idx]  # Tensor shape: (C, H, W)
#         anns = anns_batch[idx]            # List of annotations for this image
#         image_id = image_ids_batch[idx].item()

#         # Convert image tensor to numpy array and then to PIL image
#         image_np = image_tensor.numpy().astype('uint8')  # Shape: (H, W, C)
#         # Image.fromarray(image[i].numpy().astype('uint8'), mode='RGB')
#         image_pil = Image.fromarray(image_np, mode='RGB')

#         # Segment image using FastSAM
#         mask_result = segment_image_with_fastsam(image_pil)[0]
#         if mask_result is None or not hasattr(mask_result, 'masks'):
#             print(f"No masks found for image {image_id}")
#             continue
        
        
#         # visualizations
#         # image = mmcv.imread(image_np)
#         # anns_sam = {'annotations': postprocess_fastSAM(mask_result)} 
                
#         # bitmasks = []
#         # for ann in anns_sam['annotations']:
#         #     bitmasks.append(maskUtils.decode(ann['segmentation']))
            
#         # imshow_det_bboxes(
#         #         image,
#         #         bboxes=None,
#         #         labels=np.arange(len(bitmasks)),
#         #         segms=np.stack(bitmasks),
#         #         font_size=25,
#         #         show=False,
#         #         out_file=f"output_{image_id}.png"
#         # )

#         # Get predicted masks from FastSAM result
#         fs_masks = mask_result.masks.data.cpu().numpy()  # Shape: (N, H, W)

#         # Prepare ground truth masks and labels
#         gt_masks = []
#         gt_labels = []
#         width, height = image_pil.size  # Note: PIL gives (width, height)
#         for ann in anns:
#             rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
#             gt_mask = maskUtils.decode(rle)
#             gt_masks.append(gt_mask)
#             gt_labels.append(ann['category_id'])

#         # Now match predicted masks to ground truth masks
#         matched_masks = []
#         matched_labels = []
        
#         for fs_mask in fs_masks:
#             max_iou = 0
#             matched_label = None
#             for gt_mask, gt_label in zip(gt_masks, gt_labels):
#                 # Ensure masks are boolean arrays
#                 fs_mask_bool = fs_mask.astype(bool)
#                 gt_mask_bool = gt_mask.astype(bool).squeeze()
                
#                 if fs_mask_bool.shape != gt_mask_bool.shape:
#                     continue

#                 intersection = np.logical_and(fs_mask_bool, gt_mask_bool)
#                 union = np.logical_or(fs_mask_bool, gt_mask_bool)
#                 if np.sum(union) == 0:
#                     continue  # Avoid division by zero
#                 iou = np.sum(intersection) / np.sum(union)
#                 if iou > max_iou and iou > iou_threshold:
#                     max_iou = iou
#                     matched_label = gt_label
#                     matched_labels.append(matched_label)
#                     matched_masks.append(fs_mask)
                    

#         # Process matched labels and collect data
#         for fs_mask, label in zip(matched_masks, matched_labels):
#             if label is not None:
#                 # Extract object using the mask
#                 object_image = extract_object_from_mask(image_pil, fs_mask)
#                 # Extract features
#                 features = extract_features_with_dinov2(object_image)
#                 features_list.append(features)
#                 labels_list.append(label)


# # Save the features and labels as .pt files
# torch.save(features_list, 'features_list.pt')
# torch.save(labels_list, 'labels_list.pt')



import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load features and labels
features_list = torch.load('features_list.pt')
labels_list = torch.load('labels_list.pt')  

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = self.fc(x)  # Outputs raw logits
        return x 

# Prepare the dataset from features and labels
class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.stack(features).squeeze()  # Shape: [num_samples, feature_dim]
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()  # Convert labels to LongTensor
        self.labels = self.labels - 1  # Convert labels to 0-based indices (if labels start from 1)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create the dataset from the features_list and labels_list
dataset = FeaturesDataset(features_list, labels_list)

# Split dataset into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the classifier
input_dim = features_list[0].shape[1]  # Since features are 1D tensors of shape [feature_dim]
num_classes = len(set(labels_list))  # Number of unique labels

classifier = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)
classifier.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train the classifier
num_epochs = 20

for epoch in range(num_epochs):
    classifier.train()
    train_loss = 0.0
    correct = 0
    total = 0

    # Training loop
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = classifier(features)  # Outputs raw logits of shape [batch_size, num_classes]

        # Compute loss
        loss = criterion(outputs, labels) # labels are integers of shape [batch_size]
        if epoch % 10 == 0:
            print(f"loss: {loss.item()}")
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs, 1)  # Predicted class indices
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Compute average training loss and accuracy
    train_loss /= total
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Train Accuracy: {train_accuracy:.2f}%")

    # Validate the classifier
    classifier.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = classifier(features)

            # Compute loss
            loss = criterion(outputs, labels)

            # Track loss and accuracy
            val_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    # Compute average validation loss and accuracy
    val_loss /= total
    val_accuracy = 100 * correct / total

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}] Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy:.2f}%')

# Save the trained classifier
torch.save(classifier.state_dict(), 'trained_classifier.pth')

print("Training complete. Classifier saved as 'trained_classifier.pth'.")
