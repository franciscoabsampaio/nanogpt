import matplotlib.pyplot as plt
import torch


def plot_bigram(tensor: torch.tensor, encoder) -> None:
    plt.figure(figsize=(16, 16))
    plt.imshow(tensor, cmap='Blues')

    for i in range(tensor.size[0]):
        for j in range(tensor.size[1]):
            bigram = encoder.decode(i) + encoder.decode(j)
            plt.text(j, i, bigram, ha='center', va='bottom', color='gray')
            plt.text(j, i, tensor[i, j].item(), ha='center', va='top', color='gray')
    plt.axis('off')
