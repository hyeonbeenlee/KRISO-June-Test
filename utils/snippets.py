import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
from scipy.signal import butter, sosfiltfilt

def filtbutter(data, cutoff, timestep, order, mode: str = 'low'):
    f_sampling = 1 / timestep
    nyq = f_sampling * 0.5
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype=mode, analog=False, output='sos')
    data = sosfiltfilt(sos, data, axis=0)
    return data


def sec2hms(seconds):
    Hr = int(seconds // 3600)
    seconds -= Hr * 3600
    Min = int(seconds // 60)
    seconds -= Min * 60
    Sec = seconds
    return Hr, Min, Sec

def print_time(start, end):
    h, m, s = sec2hms(end - start)
    print(f"Process took {h}hrs {m}mins {s:.2f}sec.")


def df2tensor(df: pd.DataFrame):
    return torch.FloatTensor(df.to_numpy())


def path2filename(path: str):
    filefullname = os.path.basename(path)
    filename = ".".join(filefullname.split(".")[:-1])
    return filename


def conv1d_Lout(Lin, padding, dilation, kernel_size, stride):
    Lout = int((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    return Lout


def plot_template(fontsize):
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes3d.grid"] = True
    plt.rcParams["axes.xmargin"] = 0
    plt.rcParams["axes.ymargin"] = 0.2
    plt.rcParams["axes.labelsize"] = fontsize
    plt.rcParams["axes.titlesize"] = fontsize + 3
    plt.rcParams["xtick.labelsize"] = fontsize - 3
    plt.rcParams["ytick.labelsize"] = fontsize - 3
    plt.rcParams["axes.formatter.useoffset"] = False  # scientific notation off


def add_textbox(ax, string, loc: int = 3, fontsize: int = 12):
    artist = AnchoredText(string, loc=loc, prop={"fontsize": fontsize})
    ax.add_artist(artist)


def print_line():
    print(f"{'':=>150}")
    
def increase_leglw(leg, linewidth: float = 3):
    for legobj in leg.legendHandles:
        legobj.set_linewidth(linewidth)
