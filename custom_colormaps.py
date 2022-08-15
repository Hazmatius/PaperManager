import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_linearmap(cdict):
    newcmp = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=256)
    rgba = newcmp(np.linspace(0, 1, 256))
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    col = ['r', 'g', 'b']
    for xx in [0.25, 0.5, 0.75]:
        ax.axvline(xx, color='0.7', linestyle='--')
    for i in range(3):
        ax.plot(np.arange(256)/256, rgba[:, i], color=col[i])
    ax.set_xlabel('index')
    ax.set_ylabel('RGB')
    plt.show()


def _cmap_generator(*args):
    red, green, blue = list(), list(), list()
    for arg in args:
        red.append([arg[0], arg[1][0], arg[1][0]])
        green.append([arg[0], arg[1][1], arg[1][1]])
        blue.append([arg[0], arg[1][2], arg[1][2]])
    cmap_dict = {
        'red': red,
        'green': green,
        'blue': blue
    }
    return cmap_dict


def _even_cmap(*args):
    nums = np.linspace(0.0, 1.0, len(args))
    new_args = list()
    for i in range(len(args)):
        new_arg = (nums[i], args[i])
        new_args.append(new_arg)
    return _cmap_generator(*new_args)


_red = [1.0, 0.0, 0.0]
_green = [0.0, 1.0, 0.0]
_blue = [0.0, 0.0, 1.0]
_magenta = [1.0, 0.0, 1.0]
_yellow = [1.0, 1.0, 0.0]
_cyan = [0.0, 1.0, 1.0]
_black = [0.0, 0.0, 0.0]
_white = [1.0, 1.0, 1.0]
_orange = [1.0, 0.65, 0.0]


black_blue = LinearSegmentedColormap('black_blue', _even_cmap(_black, _blue))
red_black_blue = LinearSegmentedColormap('red_black_blue', _even_cmap(_red, _black, _blue))
blue_black_red = LinearSegmentedColormap('blue_black_red', _even_cmap(_blue, _black, _red))
green_black_magenta = LinearSegmentedColormap('green_black_magenta', _even_cmap(_green, _black, _magenta))
magenta_black_green = LinearSegmentedColormap('magenta_black_green', _even_cmap(_magenta, _black, _green))
green_black_orange = LinearSegmentedColormap('green_black_orange', _even_cmap(_green, _black, _orange))
orange_black_green = LinearSegmentedColormap('orange_black_green', _even_cmap(_orange, _black, _green))
cyan_black_orange = LinearSegmentedColormap('cyan_black_orange', _even_cmap(_cyan, _black, _orange))
orange_black_cyan = LinearSegmentedColormap('orange_black_cyan', _even_cmap(_orange, _black, _cyan))