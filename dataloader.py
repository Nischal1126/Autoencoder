import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

transforms = transforms.ToTensor()

data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms)

data_loader = torch.utils.data.DataLoader(dataset = data, batch_size = 8, shuffle = True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))


img = images[0].squeeze()
plt.imshow(img, cmap='gray')
plt.title(f'label: {labels[0]}')
plt.show()

