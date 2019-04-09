# coding = utf-8
# /usr/bin/env/ python

'''
Author:Ocsphy
Date:2019/3/30 0:11
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pylab import mpl

data = pd.read_excel("jd.xlsx", header=0)
print(data)

co = np.array(data['names'])
av_m = np.array(data['av_m']) * 100
gmv = np.array(data['GMV'])

gmv_grow = np.array(data['GMVgrowth']) * 100

cnames = [
    '#F0F8FF',
    '#FAEBD7',
    '#00FFFF',
    '#7FFFD4',
    '#F0FFFF',
    '#F5F5DC',
    '#FFE4C4',
    '#000000',
    '#FFEBCD',
    '#0000FF',
    '#8A2BE2',
    '#A52A2A',
    '#DEB887',
    '#5F9EA0',
    '#7FFF00',
    '#D2691E',
    '#FF7F50',
    '#6495ED',
    '#FFF8DC',
    '#DC143C']

fig, ax = plt.subplots(figsize = (9,6))

n = data.shape[0]
for i in range(n):
    ax.scatter(av_m[i], gmv_grow[i], s=gmv[i] / 10000000, c = cnames[-i],
                alpha=0.8, edgecolors='black')
    plt.scatter(av_m[i], gmv_grow[i], c = cnames[-i], edgecolors= cnames[-i] ,label = co[i])
    plt.legend(loc=0)


plt.ylabel("销售额同比增长")
plt.xlabel("平均市场规模")

mpl.rcParams['font.sans-serif'] = ['KaiTi']


fmt = '%2.0f%%'
yticks = mtick.FormatStrFormatter(fmt)

ax.yaxis.set_major_formatter(yticks)
ax.xaxis.set_major_formatter(yticks)
plt.ylim((-50, 200))
plt.xlim((-5, 60))

ax.spines['top'].set_visible(False) #去掉上边框
ax.spines['right'].set_visible(False) #去掉左边框
plt.show()


