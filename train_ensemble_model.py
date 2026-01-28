#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Ensemble Model Training for Medical Skin Disease Classification
Trains an ensemble of diverse architectures with medical-grade accuracy and uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.utils import preprocess_image
from model.ensemble.ensemble_uncertainty import EnsembleModel
from training.train_single_model import SkinDiseaseDataset, create_data_loaders, prepare_data_transforms, create_skin_lesion_datasets


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in medical datasets
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_ensemble_member(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-4, 
                         device='cuda', member_id=0, save_path='ensemble_member.pth', 
                         gradient_clipping=1.0, class_weights=None):
    """
    Train a single ensemble member with advanced medical image classification techniques
    """
    # Use focal loss for better handling of imbalanced medical data
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = FocalLoss(alpha=1, gamma=2)
    
    # Optimizer with medical-grade settings
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )
    
    model.to(device)
    
    best_val_acc = 0.0
    patience_counter = 0
    best_epoch = 0
    early_stop_patience = 7
    
    logger.info(f"Training ensemble member {member_id} for {num_epochs} epochs")
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for medical stability
            if gradient_clipping > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
            
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct_train / total_train
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels_list = []
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # Calculate validation loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels_list, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(epoch_train_loss)
        val_accuracies.append(val_acc)
        
        logger.info(f'Member {member_id}, Epoch [{epoch+1}/{num_epochs}], '
                   f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, '
                   f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save best model for this member
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
                'train_acc': epoch_train_acc
            }, save_path)
            logger.info(f"Member {member_id} - New best accuracy: {val_acc:.4f} at epoch {epoch+1}")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping for member {member_id} at epoch {epoch+1}")
            break
    
    logger.info(f"Member {member_id} best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title(f'Loss Curve - Member {member_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title(f'Accuracy Curve - Member {member_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'model/training_curves_member_{member_id}.png')
    plt.close()
    
    return best_val_acc


def create_diverse_ensemble_architecture(member_id, num_classes=7, pretrained=True):
    """
    Create different architectures for ensemble diversity with medical-grade initialization
    """
    if member_id == 0:  # Vision Transformer (ViT)
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        # Initialize head with medical-grade settings
        torch.nn.init.xavier_uniform_(model.head.weight)
        torch.nn.init.zeros_(model.head.bias)
        
    elif member_id == 1:  # ResNet50 with medical adaptations
        model = timm.create_model('resnet50', pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Initialize classifier with medical-grade settings
        torch.nn.init.xavier_uniform_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)
        
        # Add medical-grade regularization
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.01  # Lower momentum for medical stability
                
    else:  # EfficientNet-B0 with medical adaptations
        model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        # Initialize classifier with medical-grade settings
        torch.nn.init.xavier_uniform_(model.classifier.weight)
        torch.nn.init.zeros_(model.classifier.bias)
        
        # Add medical-grade regularization
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.01  # Lower momentum for medical stability
    
    return model


def calculate_class_weights(train_labels):
    """
    Calculate class weights to handle imbalanced medical datasets
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(train_labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=train_labels
    )
    return torch.FloatTensor(class_weights)


def train_ensemble_model(num_members=3, num_epochs=30, learning_rate=1e-4, 
                        device='cuda', data_dir='data/train', batch_size=8):
    """
    Train an ensemble of models with different architectures for improved medical diagnosis
    """
    logger.info(f"Training ensemble with {num_members} diverse members...")
    
    # Create datasets
    train_split, val_split = create_skin_lesion_datasets(data_dir, validation_split=0.2)
    
    if train_split is None:
        logger.error("No training data found for ensemble training. Please ensure your data is organized as data/train/class_name/images.jpg")
        return None, 0.0
    
    train_paths, train_labels = train_split
    val_paths, val_labels = val_split
    
    # Calculate class weights for imbalanced medical data
    class_weights = calculate_class_weights(train_labels)
    logger.info(f"Calculated class weights: {class_weights}")
    
    # Prepare transforms
    train_transform, val_transform = prepare_data_transforms()
    
    # Create datasets with augmentation
    train_dataset = SkinDiseaseDataset(train_paths, train_labels, transform=train_transform, augment=True)
    val_dataset = SkinDiseaseDataset(val_paths, val_labels, transform=val_transform, augment=False)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size=batch_size)
    
    ensemble_models = []
    member_accuracies = []
    
    num_classes = 7  # Adjust based on your dataset
    
    for i in range(num_members):
        logger.info(f"\nTraining ensemble member {i+1}/{num_members}: Creating architecture...")
        
        # Create model with specified architecture
        model = create_diverse_ensemble_architecture(i, num_classes, pretrained=True)
        
        # Add dropout for regularization
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.3  # Increase dropout for medical regularization
            elif isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-5  # Medical-grade batch norm stability
        
        # Train this ensemble member
        member_acc = train_ensemble_member(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            member_id=i,
            save_path=f'model/ensemble_member_{i}.pth',
            gradient_clipping=1.0,
            class_weights=class_weights.numpy() if class_weights is not None else None
        )
        
        member_accuracies.append(member_acc)
        ensemble_models.append(model)
        
        # Load the best checkpoint for this member
        try:
            checkpoint = torch.load(f'model/ensemble_member_{i}.pth', map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Member {i+1} loaded with validation accuracy: {checkpoint['val_acc']:.4f}")
        except Exception as e:
            logger.warning(f"Could not load checkpoint for member {i+1}, using current state: {e}")
    
    # Create the final ensemble model
    logger.info("\nCreating final ensemble model...")
    final_ensemble = EnsembleModel(num_classes=num_classes, n_models=len(ensemble_models))
    
    # Copy the trained weights from individual models to ensemble
    for i, trained_model in enumerate(ensemble_models):
        if i < len(final_ensemble.models):
            # Load the state dict from saved checkpoint to ensure we have the best weights
            try:
                checkpoint = torch.load(f'model/ensemble_member_{i}.pth', map_location='cpu')
                final_ensemble.models[i].load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Loaded best weights for ensemble member {i}")
            except Exception as e:
                # If checkpoint doesn't exist, copy current state
                final_ensemble.models[i].load_state_dict(trained_model.state_dict())
                logger.info(f"Copied current weights for ensemble member {i}, error: {e}")
    
    # Final evaluation of the ensemble
    final_ensemble.to(device)
    final_ensemble.eval()
    
    ensemble_preds = []
    val_labels_all = []
    ensemble_probabilities = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = final_ensemble(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            ensemble_preds.extend(predicted.cpu().numpy())
            val_labels_all.extend(labels.numpy())
            ensemble_probabilities.extend(probs.cpu().numpy())
    
    ensemble_acc = accuracy_score(val_labels_all, ensemble_preds)
    
    # Detailed evaluation metrics
    logger.info(f"\nDetailed Evaluation Results:")
    logger.info(f"Final ensemble validation accuracy: {ensemble_acc:.4f}")
    logger.info(f"Classification Report:")
    logger.info(classification_report(val_labels_all, ensemble_preds, 
                                    target_names=['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
                                                 'Nevus (Mole)', 'Dermatofibroma', 'Vascular Lesion', 'Actinic Keratosis']))
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(val_labels_all, ensemble_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
                           'Nevus (Mole)', 'Dermatofibroma', 'Vascular Lesion', 'Actinic Keratosis'],
                yticklabels=['Melanoma', 'Basal Cell Carcinoma', 'Squamous Cell Carcinoma',
                           'Nevus (Mole)', 'Dermatofibroma', 'Vascular Lesion', 'Actinic Keratosis'])
    plt.title('Confusion Matrix - Ensemble Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('model/confusion_matrix_ensemble.png')
    plt.close()
    
    # Save the ensemble model
    os.makedirs('model', exist_ok=True)
    ensemble_save_path = 'model/ensemble_skin_disease.pth'
    torch.save({
        'model_state_dict': final_ensemble.state_dict(),
        'member_accuracies': member_accuracies,
        'ensemble_accuracy': ensemble_acc,
        'member_paths': [f'model/ensemble_member_{i}.pth' for i in range(len(ensemble_models))],
        'timestamp': datetime.now().isoformat(),
        'num_classes': num_classes,
        'num_members': len(ensemble_models)
    }, ensemble_save_path)
    
    logger.info(f"Ensemble model saved to {ensemble_save_path}")
    logger.info(f"Individual member models saved as ensemble_member_0.pth, ensemble_member_1.pth, etc.")
    
    return final_ensemble, ensemble_acc


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble Model for Skin Disease Classification')
    parser.add_argument('--num_members', type=int, default=3, help='Number of ensemble members')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Training data directory')
    
    args = parser.parse_args()
    
    logger.info("Starting ensemble model training...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Train ensemble model
    ensemble_model, final_accuracy = train_ensemble_model(
        num_members=args.num_members,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        data_dir=args.data_dir,
        batch_size=args.batch_size
    )
    
    if ensemble_model is not None:
        logger.info(f"Ensemble training completed with final accuracy: {final_accuracy:.4f}")
        logger.info("Model saved as 'model/ensemble_skin_disease.pth'")
        
        # Print member accuracies
        logger.info("Individual member accuracies will be saved in the model checkpoint")
    else:
        logger.error("Ensemble training failed - no training data found.")
        logger.info("Please ensure your data is organized as data/train/class_name/images.jpg")


if __name__ == "__main__":
    main()
