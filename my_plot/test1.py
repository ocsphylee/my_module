# coding = utf-8
# /usr/bin/env/ python

'''
Author:Ocsphy
Date:2019/3/30 0:11
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as mtick


def bubles(path, modify=1000000, fontsize=15, percentage=True, **kwargs):
    data = pd.read_excel(path, header=0)
    label = np.array(data.iloc[:, 0])
    x = np.array(data.iloc[:, 1]) * 100
    y = np.array(data.iloc[:, 2]) * 100
    size = np.array(data.iloc[:, 3]) / modify
    # set limit
    if 'ylim' in kwargs.keys():
        y_lim = kwargs['ylim']
        plt.ylim((y_lim[0], y_lim[1]))
    if 'xlim' in kwargs.keys():
        x_lim = kwargs['xlim']
        plt.xlim((x_lim[0], x_lim[1]))

    n = x.shape[0]
    for i in range(n):
        ax.scatter(x[i], y[i], s=size[i],cmap="tab20b",
                   alpha=0.8, edgecolors='white')
        plt.scatter(x[i], y[i], cmap="tab20b", alpha=0.6, s=100, label=label[i])

    plt.legend(loc=0, fontsize=11,)

    # set stick
    name = data.columns.values.tolist()
    plt.xlabel(name[1], fontsize=fontsize)
    plt.ylabel(name[2], fontsize=fontsize)

    if percentage:
        # show in percentage
        fmt_x = '%2.0f%%'
        fmt_y = '%2.0f%%'
        y_ticks = mtick.FormatStrFormatter(fmt_y)
        ax.yaxis.set_major_formatter(y_ticks)
        x_ticks = mtick.FormatStrFormatter(fmt_x)
        ax.xaxis.set_major_formatter(x_ticks)

    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉左边框


if __name__ == '__main__':
    # load data
    path = "C:/Users/ASUS/Desktop/data/4.10"
    filename = "/品类.xlsx"
    # initialize picture
    fig, ax = plt.subplots(figsize=(9, 6))
    mpl.rcParams['font.sans-serif'] = ['KaiTi']

    # plot
    bubles(path+filename, modify=5000000, fontsize=15,xlim = (0, 30), ylim = (-25, 70))

    plt.show()

