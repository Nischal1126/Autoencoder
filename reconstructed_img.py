import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model import Autoencoder
import torch

data_path = "image.png"

model = Autoencoder()
model.load_state_dict(torch.load("model_weights.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

img = Image.open(data_path)
img_tensor = transform(img).unsqueeze(0)
img_tensor = img_tensor.view(1, -1)


with torch.no_grad():
    recon_img = model(img_tensor)

recon_img = recon_img.squeeze().detach().numpy().reshape(28, 28)

plt.imshow(recon_img)
plt.show()