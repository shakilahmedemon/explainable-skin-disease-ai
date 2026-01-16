import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from torchvision import transforms, datasets
from train_single_model import vit_b_16, DEVICE, NUM_CLASSES, val_dataset
import numpy as np

model_paths = [f"model/ensemble/vit_{i}.pth" for i in range(1,6)]

# Ensemble predictions
all_preds = []
all_labels = []

for model_path in model_paths:
    model = vit_b_16(pretrained=True)
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    preds = []
    labels = []

    loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            outputs = torch.softmax(model(imgs), dim=1)
            preds.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())

    all_preds.append(np.vstack(preds))
    all_labels = labels[0]  # same for all models

# Ensemble average
ensemble_probs = np.mean(all_preds, axis=0)
ensemble_preds = np.argmax(ensemble_probs, axis=1)

print("Accuracy:", accuracy_score(all_labels, ensemble_preds))
print("F1-score:", f1_score(all_labels, ensemble_preds, average='weighted'))
print("Confusion Matrix:\n", confusion_matrix(all_labels, ensemble_preds))

