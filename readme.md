# Autoencoder

A simple autoencoder implementation in PyTorch for MNIST digit reconstruction.

## Models

- **AutoencoderNN** - Fully connected neural network (784 → 4 → 784)
- **AutoencoderCNN** - Convolutional neural network with transposed convolutions

## Project Structure

```
├── model.py           # Model architectures
├── train.py           # Training script
├── dataloader.py      # MNIST data loading
├── reconstructed_img.py
└── model_weights.pth  # Saved weights
```

## Usage

```bash
# Train the model
python train.py

# Reconstruct images
python reconstructed_img.py
```

## Requirements

- PyTorch
- torchvision
