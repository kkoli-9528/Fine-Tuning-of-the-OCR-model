import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from doctr.models import crnn_vgg16_bn
import tqdm
import Levenshtein
import torch.multiprocessing as mp
import torchvision.transforms as transforms

def character_error_rate(gt, pred):
    """Compute Character Error Rate (CER)"""
    gt, pred = gt.lower(), pred.lower()
    return Levenshtein.distance(gt, pred) / len(gt) if len(gt) > 0 else 0

def word_error_rate(gt, pred):
    """Compute Word Error Rate (WER)"""
    gt_words = gt.lower().split()
    pred_words = pred.lower().split()
    return Levenshtein.distance(gt_words, pred_words) / len(gt_words) if gt_words else 0

class OCRDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_height=32):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Verify label existence
        missing_labels = []
        for f in self.image_files:
            label_path = os.path.join(self.data_dir, f.replace('.jpg', '.txt')
                                      .replace('.jpeg', '.txt')
                                      .replace('.png', '.txt'))
            if not os.path.exists(label_path):
                missing_labels.append(f)
        if missing_labels:
            raise ValueError(f"Missing labels for {len(missing_labels)} images")
        
        self.transform = transform
        self.target_height = target_height

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.txt')
                                          .replace('.jpeg', '.txt')
                                          .replace('.png', '.txt'))

        try:
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Failed to read image")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Resize to fixed height
            ratio = self.target_height / float(h)
            target_width = int(w * ratio)
            image = cv2.resize(image, (target_width, self.target_height))

            # Apply augmentations
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            # Convert to tensor and normalize
            if not isinstance(image, torch.Tensor):
                image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
                image = torch.from_numpy(image)
            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                           std=[0.229, 0.224, 0.225])
            image = normalize(image)

            # Load label
            with open(label_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()

            return image, label

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            return None, None

def get_transforms(train=False):
    transforms = []
    if train:
        transforms.extend([
            A.Affine(
                scale=(0.9, 1.1),  # Reduced augmentation strength
                rotate=(-10, 10),
                shear=(-5, 5),
                translate_percent=(-0.05, 0.05),
                p=0.7
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # Reduced blur
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.4
            ),
            A.GaussNoise(var_limit=(0.05, 0.2), p=0.2),
        ])
    
    transforms.extend([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return A.Compose(transforms)

def collate_fn(batch):
    batch = [data for data in batch if data is not None and data[0] is not None]
    if not batch:
        return torch.Tensor(), []
    
    images, labels = zip(*batch)
    
    # Dynamic padding to max width in batch
    max_width = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        pad = max_width - img.shape[2]
        padded_images.append(torch.nn.functional.pad(img, (0, pad, 0, 0), value=0))
    
    return torch.stack(padded_images), list(labels)

def initialize_model():
    original_model = crnn_vgg16_bn(pretrained=True)
    original_vocab = list(original_model.vocab)
    
    # Verify digit presence
    missing_digits = [str(i) for i in range(10) if str(i) not in original_vocab]
    if missing_digits:
        raise ValueError(f"Missing digits in pretrained vocab: {missing_digits}")
    
    # Add rupee symbol
    updated_vocab = original_vocab + ['₹']
    
    # Create new model with extended vocabulary
    model = crnn_vgg16_bn(pretrained=False, vocab=''.join(updated_vocab))
    
    # Handle size mismatch for new character
    pretrained_state = original_model.state_dict()
    
    # Copy weights for existing characters
    for layer in ['linear.weight', 'linear.bias']:
        if layer in pretrained_state:
            # Original shape: [num_original_chars, ...]
            original_weight = pretrained_state[layer]
            
            # New shape: [num_original_chars + 1, ...]
            new_weight = model.state_dict()[layer].clone()
            
            # Copy original weights for existing characters
            new_weight[:original_weight.shape[0]] = original_weight
            
            # Initialize new character weights (₹) with zeros
            pretrained_state[layer] = new_weight

    model.load_state_dict(pretrained_state, strict=False)
    return model

def train_model(train_path, val_path, num_epochs=20, batch_size=32,  # Increased epochs
                lr=1e-4, model_path="rupee_model.pt"):  # Reduced learning rate
    best_cer = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_cer': [],
        'val_wer': []
    }
    
    model = initialize_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_dataset = OCRDataset(train_path, get_transforms(True))
    val_dataset = OCRDataset(val_path, get_transforms(False))
    
    # Check for overlapping files
    train_files = set(train_dataset.image_files)
    val_files = set(val_dataset.image_files)
    overlap = train_files & val_files
    print(f"Overlapping files: {len(overlap)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for images, labels in tqdm.tqdm(train_loader, 
                                      desc=f"Epoch {epoch+1}/{num_epochs}"):
            if images.nelement() == 0:
                continue
            
            images = images.to(device)
            optimizer.zero_grad()
            
            output = model(images, target=labels)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation loop
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                if images.nelement() == 0:
                    continue
                
                images = images.to(device)
                output = model(images)
                
                # Direct prediction extraction
                raw_preds = output['preds']
                current_preds = [pred[0] for pred in raw_preds]
                
                # Confidence filtering
                filtered_preds = [
                    p if conf > 0.4 else "" 
                    for p, (_, conf) in zip(current_preds, raw_preds)
                ]
                
                all_preds.extend(filtered_preds)
                all_gts.extend(labels)
        
        # Calculate metrics
        cer, wer = 0.0, 0.0
        if len(all_gts) > 0:
            cer = sum(character_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
            wer = sum(word_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
        
        # Update tracking
        history['train_loss'].append(train_loss/len(train_loader))
        history['val_cer'].append(cer)
        history['val_wer'].append(wer)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation CER: {cer:.4f} | WER: {wer:.4f}")
        
        # Save best model
        if cer < best_cer:
            best_cer = cer
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "rupee_model_best.pt")
            print(f"New best model saved at epoch {best_epoch}")
        
        # Regular checkpoint
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    mp.freeze_support()
    train_model(
        train_path="data/train/",
        val_path="data/val/",
        num_epochs=20,
        batch_size=32,
        lr=1e-4,
        model_path="rupee_model.pt"
    )