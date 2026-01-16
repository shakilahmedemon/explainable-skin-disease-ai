import torch, random, numpy as np, os
from training.train_single_model import train_dataset, vit_base_patch16_224 as timm_model, DEVICE, BATCH_SIZE, EPOCHS, NUM_CLASSES
from torch.utils.data import DataLoader
import timm
from torch import nn, optim

NUM_MODELS = 5

os.makedirs("model/ensemble", exist_ok=True)

for i in range(1, NUM_MODELS+1):
    print(f"Training model {i}/{NUM_MODELS}")

    # Seed
    seed = 42 + i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.head = nn.Linear(model.head.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Model {i} Epoch {epoch+1} Loss: {running_loss/len(loader):.4f}")

    torch.save(model.state_dict(), f"model/ensemble/vit_{i}.pth")
    print(f"Saved vit_{i}.pth")
