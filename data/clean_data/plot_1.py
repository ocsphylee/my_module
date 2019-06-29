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


def bubles(path, sheet=None,modify=1000000, fontsize=15, percentage=True, **kwargs):
    data = pd.read_excel(path, sheet_name=sheet,header=0)
    label = np.array(data.iloc[:, 0])
    x = np.array(data.iloc[:, 1]) * 100
    y = np.array(data.iloc[:, 2]) * 100
    size = np.array(data.iloc[:, 3])
    # set limit
    if 'ylim' in kwargs.keys():
        y_lim = kwargs['ylim']
        plt.ylim((y_lim[0], y_lim[1]))
    if 'xlim' in kwargs.keys():
        x_lim = kwargs['xlim']
        plt.xlim((x_lim[0], x_lim[1]))

    n = x.shape[0]
    palette = sns.color_palette("husl", n)
    for i in range(n):
        ax.scatter(x[i], y[i], s=size[i] / modify, c=palette[i],
                   alpha=0.8, edgecolors='white')
        ax.scatter(x[i], y[i], c=palette[i], alpha=0.8, s=100,
                   edgecolors=palette[i], label=label[i])
        plt.legend(loc=0, fontsize=11)

    # set stick
    name = data.columns.values.tolist()
    plt.xlabel(name[1], fontsize=fontsize)
    plt.ylabel(name[2], fontsize=fontsize)

    if percentage:
        # show in percentage
        fmt_x = '%2.2f%%'
        fmt_y = '%2.0f%%'
        y_ticks = mtick.FormatStrFormatter(fmt_y)
        ax.yaxis.set_major_formatter(y_ticks)
        x_ticks = mtick.FormatStrFormatter(fmt_x)
        ax.xaxis.set_major_formatter(x_ticks)

    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉左边框


if __name__ == '__main__':
    # initialize picture
    fig, ax = plt.subplots(figsize=(12, 6))
    mpl.rcParams['font.sans-serif'] = ['KaiTi']

    # load data
    path = "E:\ocsphy\my_modules\data\clean_data\Tmall\\"
    filename = "2019-05-01bubble_data_Tmall"

    # plot
    name = '美妆个护'
    bubles(path + filename + ".xlsx",sheet = name, modify=0.0005, fontsize=15,xlim=(-0.25,3),ylim=(-100,570))
    plt.savefig(path + '5月\\' + filename + name +'.png', dpi=300)
    plt.show()

