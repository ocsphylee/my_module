import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



"""
h
"""


# class my_plot:
#     def __init__(self, nrows=1, ncols=1):
#         self.fig, self.axes = plt.subplots(nrows, ncols)
#
#     def plot(self, rows, cols, x, y):
#         axes = self.axes[rows][cols]
#         axes.plot(x, y)
#         return axes
#
#     def hist(self):


def stacked_bar(axes,x,y):

    """
    堆叠条形图
    输入：axes:轴对象
         x:array_like,横坐标
         y:DataFrame,需要堆叠的数据

    """
    if x.shape[0] != y.shape[0]:
        print("必须输入同样长度的数据")
        return None

    bottom = 0
    for label,col in zip(x,y.columns):

        axes.bar(
            list(range(len(x))),
            y[col],
            width=0.5,
            alpha=0.5,
            label='label',
            bottom=bottom,
            linewidth=0.8,
            # facecolor='gray',
            # edgecolor='black'
                )
        bottom +=  y[col]



if __name__ == '__main__':
    fig,axes = plt.subplots(2,1)
    data = pd.read_excel("test.xlsx", skiprows=[1], usecols=list(range(11)))
    stacked_bar(axes[0],data["公司名称"],data["PE"])


    plt.show()

