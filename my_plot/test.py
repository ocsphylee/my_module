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
    size = np.array(data.iloc[:, 3])
    # set limit
    if 'ylim' in kwargs.keys():
        lim = kwargs['ylim']
        plt.ylim((lim[0], lim[1]))
    if 'xlim' in kwargs.keys():
        lim = kwargs['ylim']
        plt.xlim((lim[0], lim[1]))

    n = x.shape[0]
    for i in range(n):
        ax.scatter(x[i], y[i], s=size[i] / modify, c=sns.color_palette("hls", n)[i],
                   alpha=0.8, edgecolors='white')
        plt.scatter(x[i], y[i], c=sns.color_palette("hls", n)[i], alpha=0.8, s=100,
                    edgecolors=sns.color_palette("hls", n)[i], label=label[i])
        plt.legend(loc=0, fontsize=fontsize)

    # set stick
    name = data.columns.values.tolist()
    plt.xlabel(name[1], fontsize=fontsize)
    plt.ylabel(name[2], fontsize=fontsize)

    if percentage:
        # show in percentage
        fmt = '%2.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)
        ax.xaxis.set_major_formatter(yticks)

    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉左边框


if __name__ == '__main__':
    # load data
    path = "C:/Users/ASUS/Desktop/files/data/医药保健.xlsx"
    # initialize picture
    fig, ax = plt.subplots(figsize=(9, 6))
    mpl.rcParams['font.sans-serif'] = ['KaiTi']

    # plot
    bubles(path, modify=5000, fontsize=15)
    # plt.savefig('./test2.jpg', dpi=300)
    plt.show()
