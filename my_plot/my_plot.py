import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class my_plot:
    def __init__(self, nrows=1, ncols=1):
        self.fig, self.axes = plt.subplots(nrows, ncols)

    def plot(self, rows, cols, x, y):
        axes = self.axes[rows][cols]
        axes.plot(x, y)
        return axes


if __name__ == '__main__':
    my_plot = my_plot(2, 2)
    x = [1, 2, 3, 4]
    y = [1, 2, 3, 4]
    axes = my_plot.plot(0, 0, x, y)
    plt.show()
