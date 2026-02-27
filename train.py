import torch
import torch.nn as nn
import torch.optim as optim
from model import Autoencoder
from dataloader import data_loader

model = Autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr= 1e-3, weight_decay= 1e-5)

epochs = 20
outputs = []

for epoch in range(epochs):
    for (img, _) in data_loader:
        img = img.reshape(-1, 28*28)
        recon = model(img)
        loss = criterion(recon, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch:{epoch + 1}, Loss: {loss.item():.4f}")
    outputs.append((epoch, img, recon))

PATH = "model_weights.pth"
torch.save(model.state_dict(), PATH)

