"""
Quick training script to create a working model fast
"""
import torch
import os
from training.train_single_model import load_pretrained_vit_model, prepare_data_transforms, create_skin_lesion_datasets, SkinDiseaseDataset, create_data_loaders
from training.train_single_model import train_model

def quick_train():
    print('Starting quick training...')
    device = torch.device('cpu')  # Use CPU for faster startup

    # Create model
    model = load_pretrained_vit_model(num_classes=7, pretrained=True)
    model_path = 'model/vit_skin_disease.pth'
    os.makedirs('model', exist_ok=True)

    # Create a minimal dataset to test
    data_dir = 'data/train'
    train_split, val_split = create_skin_lesion_datasets(data_dir, validation_split=0.2)

    if train_split is not None:
        train_paths, train_labels = train_split
        val_paths, val_labels = val_split
        
        # Use minimal transforms for speed
        train_transform, val_transform = prepare_data_transforms()
        train_dataset = SkinDiseaseDataset(train_paths, train_labels, transform=train_transform, augment=True)
        val_dataset = SkinDiseaseDataset(val_paths, val_labels, transform=val_transform, augment=False)
        
        # Small batch size for limited data
        train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, batch_size=4)
        
        # Quick training for just 3 epochs
        train_losses, val_accuracies, val_losses, best_accuracy = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=3,  # Very few epochs for quick test
            learning_rate=1e-5,  # Lower learning rate for stability
            device=device,
            early_stopping_patience=2,
            save_path=model_path
        )
        
        print(f'Quick training completed! Best accuracy: {best_accuracy:.4f}')
    else:
        print('No training data found.')

if __name__ == "__main__":
    quick_train()
