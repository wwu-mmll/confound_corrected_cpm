"""Small colour helpers shared by the plotting modules."""

import matplotlib.pyplot as plt


def get_colors_from_colormap(n_colors, colormap_name='tab10'):
    """Return n_colors distinct RGBA colors sampled from a matplotlib colormap."""
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]
    return colors
