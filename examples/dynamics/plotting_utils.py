import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_2d_traj_plot(true_pos, model_pos, plot_len=None, figsize=(5, 5), box_lim=(-2.2, 2.2)):
    """
    Comparison plot for 2d dynamical systems.
    Parameters
    ----------
    true_pos: np.array, [n_bodies, 2, traj_len]
    model_pos: np.array, [n_bodies, 2, traj_len]
    plot_len: int, if None defaults to traj_len
    figsize: tuple
    box_lim: tuple

    Returns
    -------
    fig: plt.Figure object
    """
    fig = plt.figure(figsize=figsize)
    n_bodies, space_dim, traj_len = true_pos.shape
    plot_len = traj_len if plot_len is None else plot_len
    palette = sns.color_palette("muted", n_bodies)

    ax = fig.add_subplot(111)
    for i in range(n_bodies):
        x_pos = true_pos[i, 0, :plot_len]
        y_pos = true_pos[i, 1, :plot_len]
        ax.plot(x_pos, y_pos, color=palette[i], linewidth=2)
        ax.scatter(x_pos[-1], y_pos[-1], color=palette[i], s=64)

        if model_pos is not None:
            x_pos = model_pos[i, 0, :plot_len]
            y_pos = model_pos[i, 1, :plot_len]
            ax.plot(x_pos, y_pos, color=palette[i], linestyle='--', linewidth=4)
            ax.scatter(x_pos[-1], y_pos[-1], facecolors='none', edgecolors=palette[i], s=32, linewidths=3.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y', rotation=0)
    ax.set_xlim(box_lim)
    ax.set_ylim(box_lim)

    # fake lines for legend handles
    fake_line_1 = plt.plot([0], color='black', linewidth=2, label='truth')
    fake_line_2 = plt.plot([0], color='black', linewidth=4, ls='--', label='model')
    ax.legend()
    plt.tight_layout()
    return fig


def make_traj_mse_plot(traj_mses, figsize=(8, 5), mode='median'):
    """
    Compares the average trajectory MSE vs traj_len
    Parameters
    ----------
    traj_mses: dict, { "model_name": np.array, [batch_size, traj_len] }, traj_mses['model_name'][i, t] is the
               cumulative MSE of trajectory `i` predicted by 'model_name' at time `t`.
    figsize: tuple
    mode: str, 'median' or 'mean'

    Returns
    -------
    plt.Figure object
    """
    fig = plt.figure(figsize=figsize)
    # sns.set_palette('muted')
    sns.set_palette(sns.color_palette("Set1", n_colors=len(traj_mses.keys()), desat=.75))
    ax = fig.add_subplot(111)
    for model_name, mse in sorted(traj_mses.items(), key=lambda item: item[0]):
        if mode == 'mean':
            avg_mse = mse.mean(0)
        elif mode == 'median':
            avg_mse = np.median(mse, axis=0)
        else:
            raise RuntimeError("only mean and median aggregation supported")

        ax.plot(avg_mse, linewidth=4, label=model_name)
    ax.legend()
    plt.tight_layout()
    return fig


def make_conservation_plot(traj_quants):
    """
    Parameters
    ----------
    traj_quants: dict, {
        "quantity_name": {
            "model_name": np.array, [traj_len]
        }
    }

    Returns
    -------
    plt.Figure
    """
    n_subplots = len(traj_quants.keys())
    height = n_subplots * 4
    width = height * 1.618
    fig = plt.figure(figsize=(width, height))
    # sns.set_palette('muted')

    for i, (quantity, model_data) in enumerate(traj_quants.items()):
        sns.set_palette(sns.color_palette("Set1", n_colors=len(model_data.keys()), desat=.67))
        ax = fig.add_subplot(n_subplots, 1, i+1)
        for model_name, data in sorted(model_data.items(), key=lambda item: item[0]):
            if model_name == 'truth':
                traj_len = data.shape
                ground_truth = data.mean().repeat(traj_len)  # assumes we are only plotting exactly conserved quantities
                ax.plot(ground_truth, linewidth=4, label='truth', color='black', linestyle='--')
            elif model_name == 'VOGN':
                continue
            else:
                ax.plot(data, label=model_name, linewidth=2)
        ax.set_title(quantity)
    ax.set_xlabel('t')
    ax.legend()
    plt.tight_layout()
    return fig
