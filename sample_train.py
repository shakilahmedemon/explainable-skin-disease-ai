"""
Sample training script demonstrating how to use the framework with a real dataset
This is a template that shows the expected data format and training procedure
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
import argparse
from model.vit_model import ViTModel
from model.ensemble.ensemble_uncertainty import EnsembleModel
from training.train_single_model import train_model, prepare_data_transforms
from app.utils import preprocess_image

class SkinDiseaseDataset(Dataset):
    """
    Dataset class for skin disease images
    Expected directory structure:
    dataset_root/
    ├── train/
    │   ├── melanoma/
    │   ├── basal_cell_carcinoma/
    │   ├── squamous_cell_carcinoma/
    │   ├── nevus/
    │   ├── dermatofibroma/
    │   ├── vascular_lesion/
    │   └── actinic_keratosis/
    ├── val/
    └── test/
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for fname in os.listdir(class_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, fname)
                        self.samples.append((img_path, self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_sample_dataset_splits():
    """
    Creates sample train/validation splits if real data isn't available
    This is just for demonstration purposes
    """
    print("Creating sample dataset structure...")
    
    # Create sample directories
    base_path = "data/sample_dataset"
    splits = ["train", "val", "test"]
    classes = [
        "melanoma", 
        "basal_cell_carcinoma", 
        "squamous_cell_carcinoma",
        "nevus", 
        "dermatofibroma", 
        "vascular_lesion", 
        "actinic_keratosis"
    ]
    
    for split in splits:
        split_path = os.path.join(base_path, split)
        os.makedirs(split_path, exist_ok=True)
        
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            os.makedirs(class_path, exist_ok=True)
    
    print(f"Sample dataset structure created at: {base_path}")
    print("Note: This is just the directory structure. You need to populate it with actual images.")

def train_skin_disease_model(dataset_path, model_type="vit", epochs=20, batch_size=16):
    """
    Train a skin disease classification model
    """
    print(f"Training {model_type} model for skin disease classification...")
    
    # Prepare transforms
    train_transform, val_transform = prepare_data_transforms()
    
    # Create datasets
    train_dataset = SkinDiseaseDataset(
        root_dir=os.path.join(dataset_path, "train"), 
        transform=train_transform
    )
    val_dataset = SkinDiseaseDataset(
        root_dir=os.path.join(dataset_path, "val"), 
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Create model
    num_classes = 7  # 7 skin disease classes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "vit":
        print("Using Vision Transformer model...")
        model = ViTModel(num_classes=num_classes)
    elif model_type == "ensemble":
        print("Using Ensemble model...")
        model = EnsembleModel(num_classes=num_classes, n_models=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train the model
    train_losses, val_accuracies = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        learning_rate=1e-4,
        device=device
    )
    
    # Save the trained model
    model_filename = f"skin_disease_{model_type}_model.pth"
    torch.save(model.state_dict(), os.path.join("model", model_filename))
    print(f"Model saved as {model_filename}")
    
    return model, train_losses, val_accuracies

def main():
    parser = argparse.ArgumentParser(description='Train Skin Disease Classification Model')
    parser.add_argument('--dataset-path', type=str, default='data/sample_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model-type', type=str, default='vit',
                       choices=['vit', 'ensemble'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample dataset structure')
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_dataset_splits()
        return
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_path):
        print(f"Dataset path {args.dataset_path} does not exist.")
        print("Use --create-sample-data to create a sample structure, or specify a valid path.")
        return
    
    # Train the model
    model, losses, accuracies = train_skin_disease_model(
        dataset_path=args.dataset_path,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
