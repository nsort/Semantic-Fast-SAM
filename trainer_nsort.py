import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from fastsam import FastSAM
import timm

from pycocotools import mask as maskUtils
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For FastSAM
from fastsam import FastSAM
import timm

import pycocotools.mask as maskUtils

class ZeroWasteDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.transform = transform

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
            # Convert image to numpy array
            image = np.array(image)  # Shape: (H, W, 3), dtype uint8

        return image, anns, image_id

train_dataset = ZeroWasteDataset(
    data_dir='dataset/train/data/',
    label_dir='dataset/train/'
)

def custom_collate_fn(batch):
    images, anns, image_ids = zip(*batch)
    images = list(images)  # images is a list of numpy arrays
    anns = list(anns)
    image_ids = list(image_ids)
    return images, anns, image_ids

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=custom_collate_fn
)

fast_sam_model = FastSAM('weights/FastSAM.pt')
fast_sam_model.to(device)

dinov2_model = timm.create_model('vit_base_patch16_224_dino', pretrained=True).to(device)
dinov2_model.eval()

iou_threshold = 0.5  # Define IoU threshold for matching

def segment_images_with_fastsam(images):
    mask_results = fast_sam_model(
        images,
        device=device,
        retina_masks=True,
        imgsz=[1920, 1080],
        conf=0.8,
        iou=0.8
    )    
    return mask_results

def extract_object_from_mask(image, mask):
    # Apply mask to the image
    object_image = image.copy()
    object_image[~mask] = 0
    object_image = Image.fromarray(object_image)
    return object_image

def extract_features_with_dinov2_batch(object_images):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Adjust if necessary
            std=[0.229, 0.224, 0.225]
        )
    ])

    input_tensors = [transform(img) for img in object_images]
    input_tensor = torch.stack(input_tensors).to(device)
    with torch.no_grad():
        features = dinov2_model(input_tensor)
    features = features.view(features.size(0), -1)
    return features.cpu()

# Initialize lists to collect features and labels
features_list = []
labels_list = []
counter = 0
# Main processing loop
for images_batch, anns_batch, image_ids_batch in tqdm(train_loader):
    if counter > 20:
        break
    
    counter += 1
    batch_size = len(images_batch)

    # Segment images using FastSAM
    mask_results = segment_images_with_fastsam(images_batch)

    # Initialize lists to collect object images and labels for this batch
    object_images = []
    object_labels = []

    for idx in range(batch_size):
        mask_result = mask_results[idx]
        anns = anns_batch[idx]
        image = images_batch[idx]
        image_id = image_ids_batch[idx]

        # Process mask_result and annotations
        if mask_result is None or not hasattr(mask_result, 'masks'):
            print(f"No masks found for image {image_id}")
            continue

        fs_masks = mask_result.masks.data.cpu().numpy()  # Shape: (N, H, W)

        # Prepare ground truth masks and labels
        gt_masks = []
        gt_labels = []
        height, width, _ = image.shape  # Note: image is numpy array of shape (H, W, 3)

        for ann in anns:
            rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
            gt_mask = maskUtils.decode(rle)
            gt_masks.append(gt_mask)
            gt_labels.append(ann['category_id'])

        # Now match predicted masks to ground truth masks
        for fs_mask in fs_masks:
            max_iou = 0
            matched_label = None
            for gt_mask, gt_label in zip(gt_masks, gt_labels):
                # Ensure masks are boolean arrays
                fs_mask_bool = fs_mask.astype(bool)
                gt_mask_bool = gt_mask.astype(bool).squeeze()

                if fs_mask_bool.shape != gt_mask_bool.shape:
                    continue

                intersection = np.logical_and(fs_mask_bool, gt_mask_bool)
                union = np.logical_or(fs_mask_bool, gt_mask_bool)
                if np.sum(union) == 0:
                    continue  # Avoid division by zero
                iou = np.sum(intersection) / np.sum(union)
                if iou > max_iou and iou > iou_threshold:
                    max_iou = iou
                    matched_label = gt_label

            if matched_label is not None:
                # Extract object image
                object_image = extract_object_from_mask(image, fs_mask_bool)
                object_images.append(object_image)
                object_labels.append(matched_label)

    # Process object images with DINOv2 in batch
    if object_images:
        features_batch = extract_features_with_dinov2_batch(object_images)
        features_list.append(features_batch)
        labels_list.extend(object_labels)

# Concatenate all features
features_tensor = torch.cat(features_list, dim=0)
labels_tensor = torch.tensor(labels_list)

# Save the features and labels as .pt files
torch.save(features_tensor, 'features_list_b.pt')
torch.save(labels_tensor, 'labels_list_b.pt')


# Training the classifier
# Load features and labels
features_tensor = torch.load('features_list_b.pt')  # Shape: [num_samples, feature_dim]
labels_tensor = torch.load('labels_list_b.pt')      # Shape: [num_samples]

# Prepare the dataset from features and labels
class FeaturesDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels - 1  # Convert labels to 0-based indices
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create the dataset
dataset = FeaturesDataset(features_tensor, labels_tensor)

# Split dataset into train and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_classes)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

# Initialize the classifier
input_dim = features_tensor.shape[1]  # Since features are of shape [num_samples, feature_dim]
num_classes = labels_tensor.max().item() + 1  # Number of unique labels

classifier = SimpleClassifier(input_dim=input_dim, num_classes=num_classes)
classifier.to(device)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train the classifier
num_epochs = 100

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
        loss = criterion(outputs, labels)
        
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
torch.save(classifier.state_dict(), 'trained_classifier_b.pth')

print("Training complete. Classifier saved as 'trained_classifier.pth'.")
