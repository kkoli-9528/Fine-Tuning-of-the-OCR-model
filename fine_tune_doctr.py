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
    """Compute the Character Error Rate (CER)."""
    gt, pred = gt.lower(), pred.lower()
    return Levenshtein.distance(gt, pred) / len(gt) if len(gt) > 0 else 0

def word_error_rate(gt, pred):
    """Compute the Word Error Rate (WER)."""
    gt_words, pred_words = gt.lower().split(), pred.lower().split()
    return Levenshtein.distance(" ".join(gt_words), " ".join(pred_words)) / len(gt_words) if len(gt_words) > 0 else 0

class OCRDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_height=32):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform
        self.target_height = target_height

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        label_path = os.path.join(self.data_dir, img_name.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt'))

        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Failed to read image")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize to fixed height
            h, w = image.shape[:2]
            ratio = self.target_height / h
            target_width = int(w * ratio)
            image = cv2.resize(image, (target_width, self.target_height))

            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']

            # Convert to tensor
            if not isinstance(image, torch.Tensor):
                image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
                image = torch.from_numpy(image)
            
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)

            # Load label
            with open(label_path, 'r', encoding='utf-8') as f:
                label = f.read().strip()

            return image, label

        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            return None, None

def get_transforms(train=False):
    transforms = [
        A.Affine(scale=(0.95, 1.05)),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(transforms) if train else A.Compose([transforms[-2], transforms[-1]])

def collate_fn(batch):
    batch = [data for data in batch if data is not None and data[0] is not None]
    if not batch:
        return torch.Tensor(), []
    
    images, labels = zip(*batch)
    
    # Dynamic padding to max width in batch
    max_width = max(img.shape[2] for img in images)
    padded_images = []
    for img in images:
        pad_amount = max_width - img.shape[2]
        if pad_amount > 0:
            padded_img = torch.nn.functional.pad(img, (0, pad_amount), value=0)
        else:
            padded_img = img
        padded_images.append(padded_img)
    
    return torch.stack(padded_images), list(labels)

def initialize_model():
    original_model = crnn_vgg16_bn(pretrained=True)
    original_vocab = list(original_model.vocab)
    updated_vocab = original_vocab + ['₹']
    model = crnn_vgg16_bn(pretrained=False, vocab=''.join(updated_vocab))
    pretrained_state = original_model.state_dict()
    
    # Handle different layer names across doctr versions
    for layer in ['linear.weight', 'classifier.weight', 'head.0.weight']:
        if layer in pretrained_state:
            del pretrained_state[layer]
    for layer in ['linear.bias', 'classifier.bias', 'head.0.bias']:
        if layer in pretrained_state:
            del pretrained_state[layer]
    
    model.load_state_dict(pretrained_state, strict=False)
    return model

def train_model(train_path, val_path, num_epochs=10, batch_size=32, lr=3e-4, model_path="fine_tuned_model.pt"):
    model = initialize_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_dataset = OCRDataset(train_path, get_transforms(True))
    val_dataset = OCRDataset(val_path, get_transforms(False))
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size, collate_fn=collate_fn)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = batch
            if images.nelement() == 0:
                continue
            
            images = images.to(device)
            optimizer.zero_grad()
            
            output = model(images, target=labels)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        model.eval()
        all_preds, all_gts = [], []
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                if images.nelement() == 0:
                    continue
                
                images = images.to(device)
                output = model(images)
                
                # Handle ALL output formats
                if isinstance(output['preds'][0], str):
                    # Direct string predictions (newest doctr versions)
                    all_preds.extend(output['preds'])
                else:
                    # Legacy tensor/tuple processing
                    if isinstance(output['preds'], tuple):
                        preds = output['preds'][0].cpu().numpy()
                    elif isinstance(output['preds'], list):
                        preds = [p[0].cpu().numpy() if isinstance(p, tuple) else p.cpu().numpy() 
                                for p in output['preds']]
                    else:
                        preds = output['preds'].cpu().numpy()
                    
                    decoded = [''.join([model.vocab[i] for i in pred if i < len(model.vocab)]) 
                              for pred in preds]
                    all_preds.extend(decoded)
                
                all_gts.extend(labels)
        
        if len(all_gts) > 0:
            cer = sum(character_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
            wer = sum(word_error_rate(g, p) for g,p in zip(all_gts, all_preds)) / len(all_gts)
        else:
            cer = wer = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Validation CER: {cer:.4f} | WER: {wer:.4f}\n")
        
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    mp.freeze_support()
    train_model(
        train_path="data/train/",
        val_path="data/val/",
        num_epochs=10,
        batch_size=32,
        lr=3e-4,
        model_path="rupee_model.pt"
    )