import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.models import vit_b_16  # PyTorch built-in ViT
from torch.utils.data import DataLoader

# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 10
NUM_CLASSES = 5  # adjust to your dataset
LR = 3e-4
MODEL_PATH = "model/ensemble/vit_1.pth"

# -------------------------
# Data Augmentation
# -------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
val_dataset   = datasets.ImageFolder("data/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Model
# -------------------------
model = vit_b_16(pretrained=True)
model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
model = model.to(DEVICE)

# -------------------------
# Loss and optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

