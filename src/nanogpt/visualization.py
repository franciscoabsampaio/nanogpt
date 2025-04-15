import matplotlib.pyplot as plt
import torch


def plot_bigram(tensor: torch.tensor, encoder) -> None:
    plt.figure(figsize=(16, 16))
    plt.imshow(tensor, cmap='Blues')

    for i in range(tensor.size()[0]):
        for j in range(tensor.size()[1]):
            bigram = encoder.decode(i) + encoder.decode(j)
            plt.text(j, i, bigram, ha='center', va='bottom', color='gray')
            plt.text(j, i, tensor[i, j].item(), ha='center', va='top', color='gray')
    plt.axis('off')


@torch.no_grad()
def save_plot_histogram_of_tensors(
    list_of_tensors: list[torch.Tensor],        
):
    plt.figure(figsize=(10, 6))
    legends = []
    for i, t in enumerate(list_of_tensors):
        hy, hx = torch.histogram(t.cpu(), bins=100, density=True)
        plt.plot(hx[:-1], hy)
        legends.append(f"t{i} (shape={t.shape})")
    plt.legend(legends)
    plt.savefig("output/histogram_of_tensors.png")
    plt.close()


@torch.no_grad()
def save_plot_loss(
    list_of_losses: list[float],
):
    plt.figure(figsize=(10, 5))
    legends = []
    plt.plot(list_of_losses)
    plt.legend(legends)
    plt.savefig("output/loss.png")
    plt.close()


@torch.no_grad()
def save_plot_update_to_data_ratios(
    list_of_parameters: list[torch.Tensor],
    list_of_update_to_data_ratios: list[list[float]],
):
    
    plt.figure(figsize=(10, 5))
    legends = []
    for i, p in enumerate(list_of_parameters):
        if p.ndim == 2:
            plt.plot([list_of_update_to_data_ratios[j][i] for j in range(len(list_of_update_to_data_ratios))])
            legends.append(f"p{i} (2D)")
    plt.plot([0, len(list_of_update_to_data_ratios)], [-3, -3], 'k--')
    plt.legend(legends)
    plt.savefig("output/update_to_data_ratios.png")
    plt.close()
