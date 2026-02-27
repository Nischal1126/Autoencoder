import torch
from torchvision import datasets, transforms

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

data = datasets.MNIST(root = './data', train = True, download = True, transform = transforms)

data_loader = torch.utils.data.DataLoader(dataset = data, batch_size = 8, shuffle = True)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))

print(f"Total images in dataset: {len(data)}")
print(torch.min(images), torch.max(images))

img = images[0].squeeze()


