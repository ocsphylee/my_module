# coding = utf-8
# /usr/bin/env/ python

'''
Author:Ocsphy
Date:2019/5/30 16:06
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.ticker as mtick


def load_data(path, index=None, growpby=None):
    data = pd.read_excel(path, index_col=index)
    timedata = data.groupby(growpby)
    new_data = timedata.sum()
    gmv = new_data['gmv'].unstack()
    sales = new_data['sale_qtty'].unstack()
    atv = gmv / sales
    return gmv, sales, atv


def load_cat(path, name):
    brands = pd.read_excel(path).fillna(0)
    brand_lit = brands[name].tolist()
    while 0 in brand_lit:
        brand_lit.remove(0)
    return brand_lit


def line_format(label):
    """
    Convert time label to the format of pandas line plot
    """
    if isinstance(label, pd.Period):
        label = label.to_timestamp()
    month = label.month_name()[:3] + f"-{str(label.year)[-2:]}"
    return month


def plot(data, cat, kind, stacked=True, xtick=True):
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['KaiTi']
    fig, axes = plt.subplots(figsize=(16, 10))
    data[cat].plot(
        kind=kind,
        ax=axes,
        style='o-',
        colormap='Blues_r',
        stacked=stacked)
    if xtick:
        axes.set_xticklabels(map(lambda x: line_format(x), data.index))
        plt.xticks(rotation=0)


def yoy_growth(df, period):
    tmp = df.to_period(period).groupby('dt').sum()
    tmp1 = tmp['2018']
    tmp2 = tmp['2019']
    growth = tmp2.copy()
    m, n = tmp2.shape
    for i in range(n):
        for j in range(m):
            growth.iloc[j, i] = tmp2.iloc[j, i] / tmp1.iloc[j, i] - 1
    growth.index = growth.index.to_timestamp()
    return growth


def m_growth(df, period):
    tmp = df.to_period(period).groupby('dt').sum()
    growth = tmp.copy()
    m, n = tmp.shape
    for i in range(n):
        growth.iloc[:, i] = tmp.iloc[:, i] / tmp.iloc[:, i].shift(1) - 1
    return growth


def bubles(
        x,
        y,
        size,
        label,
        modify=1000000,
        fontsize=15,
        percentage=True,
        **kwargs):
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
    # name = data.columns.values.tolist()
    # plt.xlabel(name[1], fontsize=fontsize)
    # plt.ylabel(name[2], fontsize=fontsize)

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

def bubble_data(gmv,date,kind = None,cat_gmv = None,brands = None):
    if kind == "cat":
        data = gmv.loc[date,:]
        total = data.sum()
        gmv_pct = data / total
        gmv_yoy = yoy_growth(gmv,"M").loc[date,:]
    else:
        total = cat_gmv.loc[date,kind][0]
        data = gmv[brands].loc[date,:]
        gmv_pct = data / total
        gmv_yoy = yoy_growth(gmv,"M")[brands].loc[date,:]
    bubble = gmv_pct.T
    bubble.columns = ['市场份额']
    bubble["销售额增速"] = np.array(gmv_yoy.T[date])
    bubble["销售额"] = data.T[date]
    return bubble

if __name__ == '__main__':
    gmv, sales, atv = load_data(
        'Tmall_brand.xlsx', index='dt', growpby=[
            'dt', 'main_brand_name'])
    farm = load_cat('brand_catalog.xlsx', '美妆个护')
    cat_gmv, cat_sales, cat_atv = load_data('Tmall_cat.xlsx', index='dt', growpby=['dt', 'cid1_name'])
    tmp = bubble_data(cat_gmv, '2019-04-01', kind='cat', cat_gmv=None, brands=None)
    # fig, ax = plt.subplots(figsize=(9, 6))
    # mpl.rcParams['font.sans-serif'] = ['KaiTi']
    # date = '2018-01-01'
    # bubles(atv[farm].loc[date, :], sales[farm].loc[date, :],
    #        gmv[farm].loc[date, :], farm, modify=10000, fontsize=15)
    # # plt.savefig(path + filename + '.jpg', dpi=300)
    # plt.show()
