import torch
import matplotlib.pyplot as plt
import numpy as np


def plot_disruption_predictions(out: torch.Tensor, batch, cfg):
    """Generates a matplotlib plot of disruptino predictions from a batch of network
    outputs. The plot is end-aligned so that many disruptions can be observed being
    signalled at once.

    Args:
        out (torch.Tensor): tensor of shape [batch, seq_len]
        lens: the inputted batch tuple of (x, label, lens)
        cfg: the hydra disruption plotting config
    """
    _, labels, lens = batch
    fig, ax = plt.subplots()
    n_plot = min(out.shape[0], cfg.max_plots)
    for s_idx in range(n_plot):
        s_len = lens[s_idx]
        end = out.shape[-1]
        start = end - s_len  # we plot the shots end-aligned
        sample = out[s_idx][:s_len]
        ax.plot(
            np.arange(start, end),
            sample,
            # Label non-disruptions green, disruptions red
            color="r" if labels[s_idx] == 1 else "g",
        )
    return fig
