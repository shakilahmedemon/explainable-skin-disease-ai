import torch
import random
import numpy as np
from train_single_model import DEVICE, BATCH_SIZE, EPOCHS, NUM_CLASSES, train_dataset, val_dataset, vit_b_16, nn, optim

NUM_MODELS = 5

for i in range(1, NUM_MODELS+1):
    print(f"Training model {i}/{NUM_MODELS}")

    # Set different seeds
    seed = 42 + i
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
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
        print(f"Model {i} Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), f"model/ensemble/vit_{i}.pth")
    print(f"Saved vit_{i}.pth")

