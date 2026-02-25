import matplotlib.pyplot as plt
from train import epochs, outputs

for k in range(0, epochs, 4):
    plt.figure(figsize=(14,8))
    plt.gray()
    imgs = outputs[k][1].detach().numpy()
    recon = outputs[k][2].detach().numpy()
    for i, item in enumerate(imgs):
        if i >= 10: 
            break
        plt.subplot(2, 10, i+1)
        item = item.reshape(-1, 28, 28)

        plt.imshow(item[0])

    for i, item in enumerate(recon):
        if i >=10:
            break
        plt.subplot(2, 10, i+1)
        item = item.reshape(-1, 28, 28)
        plt.imshow(item[0])