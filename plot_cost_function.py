import matplotlib.pyplot as plt
import numpy as np


class PlotCostFunction:
    count = 0

    def __init__(self, alphas):
        self.alpha = alphas
        # self.alpha = ["Rescaling", "Mean normalisation", "Standardization"]
        self.style = ['r', 'g--', 'b--', 'g--', 'b', 'r--']

    def add_function(self, x_plot, y_plot):
        plt.plot(x_plot, y_plot, self.style[self.count], label=str(self.alpha[self.count]))
        self.count += 1

    def show(self):
        plt.legend(loc='upper right')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost Function')
        # plt.yscale('log')
        # plt.yticks([10 ** 24 * (10 ** 26) ** i for i in range(11)])
        plt.grid(True)
        plt.show()
