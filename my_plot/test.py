# coding = utf-8
# /usr/bin/env/ python

'''
Author:Ocsphy
Date:2019/3/30 0:11
'''

from my_plot import my_plot
import matplotlib.pyplot as plt

if __name__ == '__main__':
    my_plot = my_plot(2, 2)
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    my_plot.plot(0, 0, x, y)
    my_plot.plot(1, 0, x, y)
    plt.show()
