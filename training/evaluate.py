import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5

val_transforms = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
val_dataset = datasets.ImageFolder("data/val", transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model_paths = [f"model/ensemble/vit_{i}.pth" for i in range(1,6)]
all_preds = []
all_labels = []

for path in model_paths:
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = torch.nn.Linear(model.head.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    preds = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = torch.softmax(model(imgs), dim=1)
            preds.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())

    all_preds.append(np.vstack(preds))
    all_labels = labels[0]

ensemble_probs = np.mean(all_preds, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

print("Accuracy:", accuracy_score(all_labels, ensemble_preds))
print("F1-score:", f1_score(all_labels, ensemble_preds, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(all_labels, ensemble_preds))
