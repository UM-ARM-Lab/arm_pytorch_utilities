import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import torch
from arm_pytorch_utilities import array_utils
from arm_pytorch_utilities.model.mdn import MixtureDensityNetwork
from matplotlib.patches import Ellipse


def clear_ax_content(ax):
    """Clear a matplotlib axis's content without changing its limits or labels"""
    for artist in ax.lines + ax.collections + ax.patches:
        artist.remove()


def confidence_ellipse(center, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse around center with covariance cov

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
                      width=ell_radius_x * 2,
                      height=ell_radius_y * 2,
                      facecolor=facecolor,
                      **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(center[0], center[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def highlight_value_ranges(discrete_array, color_map='rgbcmyk', ymin=0., ymax=1., ax=None, x_values=None):
    """Highlight the background of current figure with a label array; a value of 0 is left blank"""
    for value, start, end in array_utils.discrete_array_to_value_ranges(discrete_array):
        if value == 0:
            continue
        # use the current axis if one is not given (global call on plt module will use current axis)
        if ax is None:
            ax = plt
        if x_values is not None:
            start, end = x_values[start], x_values[end]
        ax.axvspan(start, end, facecolor=color_map[value % len(color_map)], alpha=0.3, ymin=ymin, ymax=ymax)


def plot_state(X, labels, axis_name, title):
    state_dim = X.shape[1]
    assert state_dim == len(axis_name)

    fig, axes = plt.subplots(1, state_dim, figsize=(18, 5))
    for i in range(state_dim):
        axes[i].set_xlabel(axis_name[i])
        axes[i].plot(X[:, i].numpy())
        highlight_value_ranges(labels, ax=axes[i], color_map='rr')
    fig.suptitle(title)


def plot_mdn_prediction(learned_model, X, Y, labels, axis_name, title, output_offset=2, plot_states=False,
                        sample=0):
    N = X.shape[0]
    # freeze model
    for param in learned_model.parameters():
        param.requires_grad = False

    if plot_states:
        plot_state(X, labels, axis_name, title)

    # plot output/prediction (differences)
    output_name = axis_name[output_offset:]
    output_dim = Y.shape[1]
    f2, a2 = plt.subplots(1, output_dim, figsize=(18, 5))

    pi, normal = learned_model(X)
    if sample:
        Yhat = MixtureDensityNetwork.sample(pi, normal)
        # sample multiple times
        for _ in range(1, sample):
            Yhat = torch.cat((Yhat, MixtureDensityNetwork.sample(pi, normal)), dim=0)
    else:
        Yhat = MixtureDensityNetwork.mean(pi, normal)
        stddev = MixtureDensityNetwork.stddev(pi, normal)
        stddev = stddev.numpy()
    Yhat = Yhat.numpy()

    posterior = pi.probs
    modes = np.argmax(posterior, axis=1)

    frames = np.arange(N)
    if sample:
        frames = np.tile(frames, sample)

    for i in range(output_dim):
        j = i
        a2[i].set_xlabel(output_name[i])
        a2[i].plot(Y[:, j])
        if sample:
            a2[i].scatter(frames, Yhat[:, j], alpha=0.4, color='orange')
        else:
            mean = Yhat[:, j]
            std = stddev[:, j]
            a2[i].plot(mean)
            a2[i].fill_between(frames, mean - std, mean + std, color='orange', alpha=0.2)

        highlight_value_ranges(modes, ax=a2[i], ymin=0.5)
        highlight_value_ranges(labels, ax=a2[i], color_map='rr', ymax=0.5)
    f2.suptitle(title)

    plt.figure()
    components = posterior.shape[1]
    for i in range(components):
        plt.plot(posterior[:, i])
    highlight_value_ranges(modes, ymin=0.5)
    highlight_value_ranges(labels, color_map='rr', ymax=0.5)
    plt.title('{} component posterior'.format(title))


def plot_prediction(learned_model, X, Y, labels, axis_name, title, output_offset=2, plot_states=False, sample=None):
    # ignore sample, just there to ensure consistent API with probabilistic models
    # freeze model
    for param in learned_model.parameters():
        param.requires_grad = False

    if plot_states:
        plot_state(X, labels, axis_name, title)

    # plot output/prediction (differences)
    output_name = axis_name[output_offset:]
    output_dim = Y.shape[1]
    f2, a2 = plt.subplots(1, output_dim, figsize=(18, 5))

    Yhat = learned_model(X)
    Yhat = Yhat.numpy()

    for i in range(output_dim):
        j = i
        a2[i].set_xlabel(output_name[i])
        a2[i].plot(Y[:, j])

        mean = Yhat[:, j]
        a2[i].plot(mean)
        highlight_value_ranges(labels, ax=a2[i], color_map='rr', ymax=0.5)
    f2.legend('')

    f2.suptitle(title)


def cumulative_dist(series, labels, xlabel, indicators=None):
    """
    Plot the empirical CDF of some series

    :param series: Iterable of 1D tensors (or a 2D tensor with each row being a series)
    :param labels: Names for each of the series (iterable of strings)
    :param xlabel: Name for the value the series are over
    :param indicators: Indicator variables for each data point in each series to indicate a special event
    :return: figure
    """
    assert len(series) is len(labels)
    f = plt.figure()
    for i in range(len(series)):
        x = series[i]
        label = labels[i]
        assert len(x.shape) == 1
        x = torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), x))
        N = x.shape[0]
        cumulative_ratio = np.linspace(0, 1, N)
        x, sorted_indices = torch.sort(x)
        p = plt.step(x.cpu().numpy(), cumulative_ratio, where='post', label=label)
        if indicators is not None and indicators[i] is not None:
            ind = torch.cat((torch.zeros(1, dtype=indicators[i].dtype, device=indicators[i].device), indicators[i]))
            ind = ind[sorted_indices]
            plt.plot(x[ind == 1].cpu().numpy(), cumulative_ratio[ind == 1], 'o', color=p[0].get_color(), alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel('cumulative ratio')
    plt.ylim(0, 1)
    plt.legend()
    return f
