import matplotlib.pyplot as plt
import numpy as np
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
def save_plot_loss(
    list_of_losses: list[float],
):
    plt.figure(figsize=(10, 5))
    legends = []
    plt.plot(list_of_losses)
    plt.legend(legends)
    plt.savefig("output/loss.png")
    plt.close()

def plot_lines(
    lines: list[tuple[str, list[float]]],
    *,
    title: str = None,
    baseline_y: float = None,
    outfile: str = "plot.png",
    legend_fontsize: int = 6,
    ncol_legend: int = 2,
):
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    linestyles = ['-', '--', '-.', ':']

    for i, (label, values) in enumerate(lines):
        color = colors[i % len(colors)]
        linestyle = linestyles[(i // len(colors)) % len(linestyles)]
        plt.plot(values, color=color, linestyle=linestyle, label=label)

    if baseline_y is not None:
        plt.axhline(baseline_y, color='k', linestyle='--', label=f"Baseline ({baseline_y})")

    if title:
        plt.title(title)

    plt.legend(loc='upper left', fontsize=legend_fontsize, ncol=ncol_legend)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


@torch.no_grad()
def save_plot_histogram_of_tensors(
    dict_of_tensors: dict[str, torch.Tensor],
    outfile: str = "output/histogram_of_tensors.png",
):
    lines = []
    for k, tensor in dict_of_tensors.items():
        hy, hx = torch.histogram(tensor.cpu(), bins=100, density=True)
        label = f"tensor[{k}] (shape={tuple(tensor.shape)})"
        lines.append((label, hy.tolist()))  # using hy only for line shape
    plot_lines(lines, title="Histogram of Tensors", outfile=outfile, ncol_legend=1)


@torch.no_grad()
def save_plot_update_to_data_ratios(
    dict_of_update_to_data_ratios: dict[dict[str, list[float]]],
    outfile: str = "output/update_to_data_ratios.png",
):
    lines = []
    for key, param_group in dict_of_update_to_data_ratios.items():
        for m, values in param_group.items():
            lines.append((f"p[{key}][{m}]", [max(v, -10) for v in values]))
    plot_lines(
        lines,
        title="Update-to-Data Ratios",
        baseline_y=-3,
        outfile=outfile
    )