"""Utility functions for visualization"""
import os
import pathlib

import tqdm

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

import statsmodels.nonparametric.api as smnp

import matplotlib.pyplot as plt
import seaborn as sns

#from calibre.calibration import coverage

#import calibre.util.metric as metric_util

from matplotlib.colors import BoundaryNorm

def posterior_heatmap_2d(plot_data, X,
                         X_monitor=None,
                         cmap='inferno_r',
                         norm=None, norm_method="percentile",
                         save_addr=''):
    """Plots colored 2d heatmap using scatterplot.

    Args:
        plot_data: (np.ndarray) plot data whose color to visualize over
            2D surface, shape (N, ).
        X: (np.ndarray) locations of the plot data, shape (N, 2).
        X_monitor: (np.ndarray or None) Locations to plot data points to.
        cmap: (str) Name of color map.
        norm: (BoundaryNorm or None) Norm values to adjust color map.
            If None then a new norm will be created according to norm_method.
        norm_method: (str) The name of method to compute norm values.
            See util.visual.make_color_norm for detail.
        save_addr: (str) Address to save image to.

    Returns:
        (matplotlib.colors.BoundaryNorm) A color norm object for color map
            to be passed to a matplotlib.pyplot function.
    """
    if save_addr:
        pathlib.Path(save_addr).parent.mkdir(parents=True, exist_ok=True)
        plt.ioff()

    if not norm:
        norm = make_color_norm(plot_data, method=norm_method)

    # 2d color plot using scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(x=X[:, 0], y=X[:, 1],
                s=3,
                c=plot_data, cmap=cmap, norm=norm)
    cbar = plt.colorbar()

    # plot monitors
    if isinstance(X_monitor, np.ndarray):
        plt.scatter(x=X_monitor[:, 0], y=X_monitor[:, 1],
                    s=10, c='black')

    # adjust plot window
    plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))

    if save_addr:
        plt.savefig(save_addr, bbox_inches='tight')
        plt.close()
        plt.ion()
    else:
        plt.show()

    return norm
