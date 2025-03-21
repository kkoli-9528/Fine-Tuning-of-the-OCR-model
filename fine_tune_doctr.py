# Attempt No: 1

# import os
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset, DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from doctr.models import crnn_vgg16_bn
# import tqdm
# import Levenshtein
# import torch.multiprocessing as mp
# import torchvision.transforms as transforms

# def character_error_rate(gt, pred):
#     """Compute Character Error Rate (CER)"""
#     gt, pred = gt.lower(), pred.lower()
#     return Levenshtein.distance(gt, pred) / len(gt) if len(gt) > 0 else 0

# def word_error_rate(gt, pred):
#     """Compute Word Error Rate (WER)"""
#     gt_words = gt.lower().split()
#     pred_words = pred.lower().split()
#     return Levenshtein.distance(gt_words, pred_words) / len(gt_words) if gt_words else 0

# def verify_vocab(model):
#     """Verify model vocabulary composition"""
#     print("\n=== Model Vocabulary Verification ===")
#     print(f"Vocabulary: {model.vocab}")
#     print(f"₹ in vocabulary: {'₹' in model.vocab}")
#     print(f"Digits present: {all(str(i) in model.vocab for i in range(10))}")
#     print(f"Total characters: {len(model.vocab)}")
#     print("====================================\n")

# class OCRDataset(Dataset):
#     def __init__(self, data_dir, transform=None, target_height=32):
#         self.data_dir = data_dir
#         self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
#         # Verify label existence
#         missing_labels = []
#         for f in self.image_files:
#             label_path = os.path.join(self.data_dir, f.replace('.jpg', '.txt')
#                                       .replace('.jpeg', '.txt')
#                                       .replace('.png', '.txt'))
#             if not os.path.exists(label_path):
#                 missing_labels.append(f)
#         if missing_labels:
#             raise ValueError(f"Missing labels for {len(missing_labels)} images")
        
#         self.transform = transform
#         self.target_height = target_height

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.data_dir, img_name)
#         label_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.txt')
#                                           .replace('.jpeg', '.txt')
#                                           .replace('.png', '.txt'))

#         try:
#             # Load and preprocess image
#             image = cv2.imread(img_path)
#             if image is None:
#                 raise ValueError("Failed to read image")
            
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             h, w = image.shape[:2]
            
#             # Resize to fixed height
#             ratio = self.target_height / float(h)
#             target_width = int(w * ratio)
#             image = cv2.resize(image, (target_width, self.target_height))

#             # Apply augmentations
#             if self.transform:
#                 transformed = self.transform(image=image)
#                 image = transformed['image']

#             # Convert to tensor and normalize
#             if not isinstance(image, torch.Tensor):
#                 image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
#                 image = torch.from_numpy(image)
            
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                            std=[0.229, 0.224, 0.225])
#             image = normalize(image)

#             # Load label
#             with open(label_path, 'r', encoding='utf-8') as f:
#                 label = f.read().strip()

#             return image, label

#         except Exception as e:
#             print(f"Error processing {img_name}: {str(e)}")
#             return None, None

# def get_transforms(train=False):
#     transforms = []
#     if train:
#         transforms.extend([
#             A.Affine(
#                 scale=(0.9, 1.1),
#                 rotate=(-10, 10),
#                 shear=(-5, 5),
#                 translate_percent=(-0.05, 0.05),
#                 p=0.7
#             ),
#             A.GaussianBlur(blur_limit=(3, 5), p=0.2),
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.2, 
#                 contrast_limit=0.2, 
#                 p=0.4
#             ),
#             A.GaussNoise(var_limit=(0.05, 0.2), p=0.2),
#         ])
    
#     transforms.extend([
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])
#     return A.Compose(transforms)

# def collate_fn(batch):
#     batch = [data for data in batch if data is not None and data[0] is not None]
#     if not batch:
#         return torch.Tensor(), []
    
#     images, labels = zip(*batch)
    
#     # Dynamic padding to max width in batch
#     max_width = max(img.shape[2] for img in images)
#     padded_images = []
#     for img in images:
#         pad = max_width - img.shape[2]
#         padded_images.append(torch.nn.functional.pad(img, (0, pad, 0, 0), value=0))
    
#     return torch.stack(padded_images), list(labels)

# def initialize_model():
#     original_model = crnn_vgg16_bn(pretrained=True)
#     original_vocab = list(original_model.vocab)
    
#     # Verify digit presence
#     missing_digits = [str(i) for i in range(10) if str(i) not in original_vocab]
#     if missing_digits:
#         raise ValueError(f"Missing digits in pretrained vocab: {missing_digits}")
    
#     # Add rupee symbol
#     updated_vocab = original_vocab + ['₹']
    
#     # Create new model with extended vocabulary
#     model = crnn_vgg16_bn(pretrained=False, vocab=''.join(updated_vocab))
    
#     # Handle size mismatch for new character
#     pretrained_state = original_model.state_dict()
    
#     # Copy weights for existing characters
#     for layer in ['linear.weight', 'linear.bias']:
#         if layer in pretrained_state:
#             original_weight = pretrained_state[layer]
#             new_weight = model.state_dict()[layer].clone()
#             new_weight[:original_weight.shape[0]] = original_weight
#             pretrained_state[layer] = new_weight

#     model.load_state_dict(pretrained_state, strict=False)
#     return model

# def train_model(train_path, val_path, num_epochs=20, batch_size=32, 
#                 lr=1e-4, model_path="rupee_model.pt"):
#     best_cer = float('inf')
#     best_epoch = 0
#     history = {
#         'train_loss': [],
#         'val_cer': [],
#         'val_wer': []
#     }
    
#     model = initialize_model()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     # Initial vocabulary check
#     verify_vocab(model)
    
#     train_dataset = OCRDataset(train_path, get_transforms(True))
#     val_dataset = OCRDataset(val_path, get_transforms(False))
    
#     # Check for overlapping files
#     train_files = set(train_dataset.image_files)
#     val_files = set(val_dataset.image_files)
#     overlap = train_files & val_files
#     print(f"Overlapping files: {len(overlap)}")
    
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size*2,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
        
#         # Training loop
#         for images, labels in tqdm.tqdm(train_loader, 
#                                       desc=f"Epoch {epoch+1}/{num_epochs}"):
#             if images.nelement() == 0:
#                 continue
            
#             images = images.to(device)
#             optimizer.zero_grad()
            
#             output = model(images, target=labels)
#             loss = output['loss']
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         # Validation loop
#         model.eval()
#         all_preds, all_gts = [], []
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 if images.nelement() == 0:
#                     continue
                
#                 images = images.to(device)
#                 output = model(images)
                
#                 # Direct prediction extraction
#                 raw_preds = output['preds']
#                 current_preds = [pred[0] for pred in raw_preds]
                
#                 # Confidence filtering
#                 filtered_preds = [
#                     p if conf > 0.4 else "" 
#                     for p, (_, conf) in zip(current_preds, raw_preds)
#                 ]
                
#                 all_preds.extend(filtered_preds)
#                 all_gts.extend(labels)
        
#         # Calculate metrics
#         cer, wer = 0.0, 0.0
#         if len(all_gts) > 0:
#             cer = sum(character_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
#             wer = sum(word_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
        
#         # Update tracking
#         history['train_loss'].append(train_loss/len(train_loader))
#         history['val_cer'].append(cer)
#         history['val_wer'].append(wer)
        
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print(f"Train Loss: {train_loss/len(train_loader):.4f}")
#         print(f"Validation CER: {cer:.4f} | WER: {wer:.4f}")
        
#         # Save best model
#         if cer < best_cer:
#             best_cer = cer
#             best_epoch = epoch + 1
#             torch.save(model.state_dict(), "rupee_model_best.pt")
#             print(f"New best model saved at epoch {best_epoch}")
        
#         # Regular checkpoint
#         torch.save(model.state_dict(), model_path)

#     # Final verification
#     print("\n=== Final Model Check ===")
#     verify_vocab(model)
    
#     # Check saved checkpoint
#     print("Checking saved checkpoint...")
#     checkpoint = torch.load("rupee_model_best.pt")
#     loaded_model = crnn_vgg16_bn(pretrained=False, vocab=model.vocab)
#     loaded_model.load_state_dict(checkpoint)
#     verify_vocab(loaded_model)

# def test_preprocessing(image_path):
#     """Verify image preprocessing pipeline"""
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Match training preprocessing
#     h, w = image.shape[:2]
#     ratio = 32 / float(h)
#     target_width = int(w * ratio)
#     resized = cv2.resize(image, (target_width, 32))
    
#     # Normalization
#     transform = get_transforms(train=False)
#     processed = transform(image=resized)['image']
    
#     # Visualization
#     cv2.imwrite("preprocessing_check.jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
#     print("Input shape:", processed.shape)
#     return processed.unsqueeze(0)

# def predict(image_path, model_path="rupee_model_best.pt"):
#     """Run inference on a single image"""
#     # Initialize model with verified vocabulary
#     original_model = crnn_vgg16_bn(pretrained=True)
#     original_vocab = list(original_model.vocab)
#     model = crnn_vgg16_bn(pretrained=False, vocab=''.join(original_vocab + ['₹']))
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
    
#     # Verify loaded vocab
#     verify_vocab(model)
    
#     # Preprocess image
#     input_tensor = test_preprocessing(image_path)
    
#     # Predict
#     with torch.no_grad():
#         output = model(input_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))
#         pred = output['preds'][0][0]  # (text, confidence)
    
#     print(f"\nPrediction: {pred[0]} | Confidence: {pred[1]:.2f}")
#     return pred[0]

# if __name__ == "__main__":
#     mp.freeze_support()
    
#     # Train the model
#     train_model(
#         train_path="data/train/",
#         val_path="data/val/",
#         num_epochs=20,
#         batch_size=32,
#         lr=1e-4,
#         model_path="rupee_model.pt"
#     )
    
#     # Test inference
#     test_image = "test_image.jpg"  # Replace with your image path
#     if os.path.exists(test_image):
#         prediction = predict(test_image)
#         print(f"\nFinal Prediction for {test_image}: {prediction}")

# --------------------------------------------------------------------------------------------

# Attempt No: 2

# import os
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset, DataLoader
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from doctr.models import crnn_vgg16_bn
# import tqdm
# import Levenshtein
# import torch.multiprocessing as mp
# import torchvision.transforms as transforms

# def character_error_rate(gt, pred):
#     """Compute Character Error Rate (CER)"""
#     gt, pred = gt.lower(), pred.lower()
#     return Levenshtein.distance(gt, pred) / len(gt) if len(gt) > 0 else 0

# def word_error_rate(gt, pred):
#     """Compute Word Error Rate (WER)"""
#     gt_words = gt.lower().split()
#     pred_words = pred.lower().split()
#     return Levenshtein.distance(gt_words, pred_words) / len(gt_words) if gt_words else 0

# def verify_vocab(model):
#     """Verify model vocabulary composition"""
#     print("\n=== Model Vocabulary Verification ===")
#     print(f"Vocabulary: {model.vocab}")
#     print(f"₹ in vocabulary: {'₹' in model.vocab}")
#     print(f"Digits present: {all(str(i) in model.vocab for i in range(10))}")
#     print(f"Total characters: {len(model.vocab)}")
#     print("====================================\n")

# class OCRDataset(Dataset):
#     def __init__(self, data_dir, transform=None, target_height=32):
#         self.data_dir = data_dir
#         self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
#         # Verify label existence
#         missing_labels = []
#         for f in self.image_files:
#             label_path = os.path.join(self.data_dir, f.replace('.jpg', '.txt')
#                                       .replace('.jpeg', '.txt')
#                                       .replace('.png', '.txt'))
#             if not os.path.exists(label_path):
#                 missing_labels.append(f)
#         if missing_labels:
#             raise ValueError(f"Missing labels for {len(missing_labels)} images")
        
#         self.transform = transform
#         self.target_height = target_height

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img_name = self.image_files[idx]
#         img_path = os.path.join(self.data_dir, img_name)
#         label_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.txt')
#                                           .replace('.jpeg', '.txt')
#                                           .replace('.png', '.txt'))

#         try:
#             # Load and preprocess image
#             image = cv2.imread(img_path)
#             if image is None:
#                 raise ValueError("Failed to read image")
            
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             h, w = image.shape[:2]
            
#             # Resize to fixed height
#             ratio = self.target_height / float(h)
#             target_width = int(w * ratio)
#             image = cv2.resize(image, (target_width, self.target_height))

#             # Apply augmentations
#             if self.transform:
#                 transformed = self.transform(image=image)
#                 image = transformed['image']

#             # Convert to tensor and normalize
#             if not isinstance(image, torch.Tensor):
#                 image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
#                 image = torch.from_numpy(image)
            
#             normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
#                                            std=[0.229, 0.224, 0.225])
#             image = normalize(image)

#             # Load label
#             with open(label_path, 'r', encoding='utf-8') as f:
#                 label = f.read().strip()

#             return image, label

#         except Exception as e:
#             print(f"Error processing {img_name}: {str(e)}")
#             return None, None

# def get_transforms(train=False):
#     transforms = []
#     if train:
#         transforms.extend([
#             A.Affine(
#                 scale=(0.9, 1.1),
#                 rotate=(-10, 10),
#                 shear=(-5, 5),
#                 translate_percent=(-0.05, 0.05),
#                 p=0.7
#             ),
#             A.GaussianBlur(blur_limit=(3, 5), p=0.2),
#             A.RandomBrightnessContrast(
#                 brightness_limit=0.2, 
#                 contrast_limit=0.2, 
#                 p=0.4
#             ),
#             A.GaussNoise(var_limit=(0.05, 0.2), p=0.2),
#         ])
    
#     transforms.extend([
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ToTensorV2()
#     ])
#     return A.Compose(transforms)

# def collate_fn(batch):
#     batch = [data for data in batch if data is not None and data[0] is not None]
#     if not batch:
#         return torch.Tensor(), []
    
#     images, labels = zip(*batch)
    
#     # Dynamic padding to max width in batch
#     max_width = max(img.shape[2] for img in images)
#     padded_images = []
#     for img in images:
#         pad = max_width - img.shape[2]
#         padded_images.append(torch.nn.functional.pad(img, (0, pad, 0, 0), value=0))
    
#     return torch.stack(padded_images), list(labels)

# def initialize_model():
#     original_model = crnn_vgg16_bn(pretrained=True)
#     original_vocab = list(original_model.vocab)
    
#     # Verify digit presence
#     missing_digits = [str(i) for i in range(10) if str(i) not in original_vocab]
#     if missing_digits:
#         raise ValueError(f"Missing digits in pretrained vocab: {missing_digits}")
    
#     # Add rupee symbol
#     updated_vocab = original_vocab + ['₹']
    
#     # Create new model with extended vocabulary
#     model = crnn_vgg16_bn(pretrained=False, vocab=''.join(updated_vocab))
    
#     # Handle size mismatch for new character
#     pretrained_state = original_model.state_dict()
    
#     # Copy weights for existing characters
#     for layer in ['linear.weight', 'linear.bias']:
#         if layer in pretrained_state:
#             original_weight = pretrained_state[layer]
#             new_weight = model.state_dict()[layer].clone()
#             new_weight[:original_weight.shape[0]] = original_weight
#             pretrained_state[layer] = new_weight

#     model.load_state_dict(pretrained_state, strict=False)
#     return model

# def train_model(train_path, val_path, num_epochs=20, batch_size=32, 
#                 lr=1e-4, model_path="rupee_model.pt"):
#     best_cer = float('inf')
#     best_epoch = 0
#     history = {
#         'train_loss': [],
#         'val_cer': [],
#         'val_wer': []
#     }
    
#     model = initialize_model()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     # Initial vocabulary check
#     verify_vocab(model)
    
#     train_dataset = OCRDataset(train_path, get_transforms(True))
#     val_dataset = OCRDataset(val_path, get_transforms(False))
    
#     # Check for overlapping files
#     train_files = set(train_dataset.image_files)
#     val_files = set(val_dataset.image_files)
#     overlap = train_files & val_files
#     print(f"Overlapping files: {len(overlap)}")
    
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=batch_size, 
#         shuffle=True,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=batch_size*2,
#         collate_fn=collate_fn,
#         num_workers=4,
#         pin_memory=True
#     )
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
        
#         # Training loop
#         for images, labels in tqdm.tqdm(train_loader, 
#                                       desc=f"Epoch {epoch+1}/{num_epochs}"):
#             if images.nelement() == 0:
#                 continue
            
#             images = images.to(device)
#             optimizer.zero_grad()
            
#             output = model(images, target=labels)
#             loss = output['loss']
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         # Validation loop
#         model.eval()
#         all_preds, all_gts = [], []
#         with torch.no_grad():
#             for images, labels in val_loader:
#                 if images.nelement() == 0:
#                     continue
                
#                 images = images.to(device)
#                 output = model(images)
                
#                 # Direct prediction extraction
#                 raw_preds = output['preds']
#                 current_preds = [pred[0] for pred in raw_preds]
                
#                 # Confidence filtering
#                 filtered_preds = [
#                     p if conf > 0.4 else "" 
#                     for p, (_, conf) in zip(current_preds, raw_preds)
#                 ]
                
#                 all_preds.extend(filtered_preds)
#                 all_gts.extend(labels)
        
#         # Calculate metrics
#         cer, wer = 0.0, 0.0
#         if len(all_gts) > 0:
#             cer = sum(character_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
#             wer = sum(word_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
        
#         # Update tracking
#         history['train_loss'].append(train_loss/len(train_loader))
#         history['val_cer'].append(cer)
#         history['val_wer'].append(wer)
        
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print(f"Train Loss: {train_loss/len(train_loader):.4f}")
#         print(f"Validation CER: {cer:.4f} | WER: {wer:.4f}")
        
#         # Save best model
#         if cer < best_cer:
#             best_cer = cer
#             best_epoch = epoch + 1
#             torch.save(model.state_dict(), "rupee_model_best.pt")
#             print(f"New best model saved at epoch {best_epoch}")
        
#         # Regular checkpoint
#         torch.save(model.state_dict(), model_path)

#     # Final verification
#     print("\n=== Final Model Check ===")
#     verify_vocab(model)
    
#     # Check saved checkpoint
#     print("Checking saved checkpoint...")
#     checkpoint = torch.load("rupee_model_best.pt", map_location=device)
#     loaded_model = crnn_vgg16_bn(pretrained=False, vocab=model.vocab).to(device)
#     loaded_model.load_state_dict(checkpoint)
#     verify_vocab(loaded_model)

# def test_preprocessing(image_path):
#     """Verify image preprocessing pipeline"""
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     # Match training preprocessing
#     h, w = image.shape[:2]
#     ratio = 32 / float(h)
#     target_width = int(w * ratio)
#     resized = cv2.resize(image, (target_width, 32))
    
#     # Normalization
#     transform = get_transforms(train=False)
#     processed = transform(image=resized)['image']
    
#     # Visualization
#     cv2.imwrite("preprocessing_check.jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
#     print("Input shape:", processed.shape)
#     return processed.unsqueeze(0)

# def predict(image_path, model_path="rupee_model_best.pt"):
#     """Run inference on a single image"""
#     # Initialize device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Initialize model with verified vocabulary
#     original_model = crnn_vgg16_bn(pretrained=True)
#     original_vocab = list(original_model.vocab)
#     model = crnn_vgg16_bn(pretrained=False, vocab=''.join(original_vocab + ['₹'])).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
    
#     # Verify loaded vocab
#     verify_vocab(model)
    
#     # Preprocess image
#     input_tensor = test_preprocessing(image_path).to(device)
    
#     # Predict
#     with torch.no_grad():
#         output = model(input_tensor)
#         # Handle different prediction formats
#         if isinstance(output['preds'], list) and len(output['preds']) > 0:
#             # New format: list of tuples (prediction, confidence)
#             pred, confidence = output['preds'][0]
#         else:
#             # Fallback to string prediction
#             pred = output['preds'][0]
#             confidence = 1.0  # Default confidence
    
#     print(f"\nPrediction: {pred} | Confidence: {confidence:.2f}")
#     return pred

# if __name__ == "__main__":
#     mp.freeze_support()
    
#     # Train the model
#     train_model(
#         train_path="data/train/",
#         val_path="data/val/",
#         num_epochs=20,
#         batch_size=32,
#         lr=1e-4,
#         model_path="rupee_model.pt"
#     )
    
#     # Test inference
#     test_image = "test_image.jpg"  # Replace with your image path
#     if os.path.exists(test_image):
#         prediction = predict(test_image)
#         print(f"\nFinal Prediction for {test_image}: {prediction}")

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from doctr.models import crnn_vgg16_bn, db_resnet50
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

def verify_vocab(model):
    """Verify model vocabulary composition"""
    print("\n=== Model Vocabulary Verification ===")
    print(f"Vocabulary: {model.vocab}")
    print(f"₹ in vocabulary: {'₹' in model.vocab}")
    print(f"Digits present: {all(str(i) in model.vocab for i in range(10))}")
    print(f"Total characters: {len(model.vocab)}")
    print("====================================\n")

class OCRDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_height=32):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Verify label existence
        missing_labels = []
        for f in self.image_files:
            label_path = os.path.join(self.data_dir, f.replace('.jpg', '.txt'))
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
        label_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.txt'))

        try:
            # Load and preprocess image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Failed to read image")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            
            # Resize to fixed height with minimum width
            ratio = self.target_height / float(h)
            min_width = 64  # Prevent overly narrow images
            target_width = max(int(w * ratio), min_width)
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
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                shear=(-5, 5),
                translate_percent=(-0.05, 0.05),
                p=0.7
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
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
            original_weight = pretrained_state[layer]
            new_weight = model.state_dict()[layer].clone()
            new_weight[:original_weight.shape[0]] = original_weight
            pretrained_state[layer] = new_weight

    model.load_state_dict(pretrained_state, strict=False)
    return model

def train_model(train_path, val_path, num_epochs=50, batch_size=32, 
                lr=1e-4, model_path="rupee_model.pt"):
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
    
    # Initial vocabulary check
    verify_vocab(model)
    
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
    
    # Separate parameters for CNN and RNN
    cnn_params = []
    rnn_params = []
    for name, param in model.named_parameters():
        if 'feat_extractor' in name:  # CNN parameters
            cnn_params.append(param)
        else:  # RNN and linear layer parameters
            rnn_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': cnn_params, 'lr': 1e-5},  # Lower LR for CNN
        {'params': rnn_params, 'lr': 1e-4}   # Higher LR for RNN and linear layers
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
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
                
                # Dynamic confidence thresholding
                confidences = [conf for _, conf in raw_preds]
                avg_confidence = np.mean(confidences) if confidences else 0.0
                threshold = max(0.2, avg_confidence - 0.1)  # Adaptive threshold
                
                current_preds = [
                    p if conf > threshold else ""
                    for p, (_, conf) in zip([pred[0] for pred in raw_preds], raw_preds)
                ]
                
                all_preds.extend(current_preds)
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

    # Final verification
    print("\n=== Final Model Check ===")
    verify_vocab(model)
    
    # Check saved checkpoint
    print("Checking saved checkpoint...")
    checkpoint = torch.load("rupee_model_best.pt", map_location=device)
    loaded_model = crnn_vgg16_bn(pretrained=False, vocab=model.vocab).to(device)
    loaded_model.load_state_dict(checkpoint)
    verify_vocab(loaded_model)

def detect_text_regions(image_path):
    """Detect text regions using doctr's text detection model"""
    det_model = db_resnet50(pretrained=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    det_model.to(device)
    
    # Read and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    img_tensor = transform(image=img)['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Forward pass
    with torch.no_grad():
        seg_map = det_model(img_tensor)
    
    # Simple post-processing (replace with actual box extraction)
    h, w = img.shape[:2]
    return [(0, 0, w, h)]  # Temporary placeholder

def preprocess(image):
    """Preprocess image for inference"""
    h, w = image.shape[:2]
    ratio = 32 / float(h)
    min_width = 64
    target_width = max(int(w * ratio), min_width)
    resized = cv2.resize(image, (target_width, 32))
    
    transform = get_transforms(train=False)
    processed = transform(image=resized)['image']
    return processed.unsqueeze(0)

def predict(image_path, model_path="rupee_model_best.pt"):
    """Run inference on a single image"""
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with verified vocabulary
    original_model = crnn_vgg16_bn(pretrained=True)
    original_vocab = list(original_model.vocab)
    model = crnn_vgg16_bn(pretrained=False, vocab=''.join(original_vocab + ['₹'])).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Verify loaded vocab
    verify_vocab(model)
    
    # Detect text regions
    text_boxes = detect_text_regions(image_path)
    if not text_boxes:
        print("No text regions detected.")
        return ""
    
    # Predict for each region
    predictions = []
    for box in text_boxes:
        x1, y1, x2, y2 = box
        cropped_img = cv2.imread(image_path)[y1:y2, x1:x2]
        processed = preprocess(cropped_img).to(device)
        
        with torch.no_grad():
            output = model(processed)
            pred, confidence = output['preds'][0]
            predictions.append((pred, confidence))
    
    # Combine predictions
    final_pred = " ".join([p for p, _ in predictions if p])
    avg_confidence = np.mean([c for _, c in predictions]) if predictions else 0.0
    
    print(f"\nPrediction: {final_pred} | Confidence: {avg_confidence:.2f}")
    return final_pred

if __name__ == "__main__":
    mp.freeze_support()
    
    # Train the model
    train_model(
        train_path="data/train/",
        val_path="data/val/",
        num_epochs=50,
        batch_size=32,
        lr=1e-4,
        model_path="rupee_model.pt"
    )
    
    # Test inference
    test_image = "test_image.jpg"  # Replace with your image path
    if os.path.exists(test_image):
        prediction = predict(test_image)
        print(f"\nFinal Prediction for {test_image}: {prediction}")