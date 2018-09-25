from time import time
from numpy.linalg import pinv

import numpy as np

from plot_cost_function import PlotCostFunction


class LR_NEW:
    """
    Implementação melhorada de regressão linear - com gradient descent e normal equation
    """
    B, X, y, m, n = None, None, None, None, None
    x_plot, y_plot, plot_obj = None, None, None 


    def __init__(self, alphas=[.2]):
        self.alphas = alphas 
        self.plot_obj = PlotCostFunction(self.alphas)


    def init_lr(self):
        """
        Inicializa os valores da classe
        """
        self.m = self.X.shape[0]
        self.n = self.X.shape[1]

        self.B = np.zeros(self.n)

        self.x_plot = []
        self.y_plot = []


    def fit(self, x, y, use_gd=True, iter=10000, tol=.01):
        """
        Realiza o treinamento do modelo

        Parameters
        ----------
        x     : array_like
            features.
        y     : array_like
            targets
        use_gd:
            if True será utilizado o GD, caso contrario, NE
        iter  :
            Numero máximo de iterações
        tol   :
            Tolerancia

        Returns
        -------
        out : ndarray
            Retrona o custo
        """

        self.X = x
        self.y = y

        self.X = np.append(np.ones((len(self.X), 1)), self.X, axis=1)
        self.init_lr()

        if use_gd:
            for alpha in self.alphas:
                self.init_lr()
                self.gradient_descent(alpha, iter, tol)
        else:
            self.normal_equation()

        return self.cost_function()


    def predict(self, x_test):
        """
        Realiza a predição de todo o x_test 

        Parameters
        ----------
        x_test : array_like
            vetor de features.

        Returns
        -------
        out : ndarray
           Um vetor com os valores interpolados
        """
        x_test = np.append(np.ones((len(x_test), 1)), x_test, axis=1)
        return x_test.dot(self.B)


    def cost_function(self):
        """
        Calcula o valor da função de custo de todos armazenados no treinamento 

        Returns
        -------
        out : long
           valor do custo
        """
        diff = np.matrix(self.X.dot(self.B)).transpose() - self.y
        diff = np.asarray(diff).reshape(-1)
        return np.sum(diff ** 2) / (2 * self.m)


    def gradient_descent(self, alpha, iterations, tol):
        """
        Implementação do gradient descent
        """
        for iteration in range(iterations):
            # Hypothesis Values
            h = np.matrix(self.X.dot(self.B)).transpose()
            # Difference b/w Hypothesis and Actual Y
            loss = h - self.y
            # Gradient Calculation
            gradient = self.X.T.dot(loss) / self.m
            # Changing Values of B using Gradient
            self.B = (np.matrix(self.B).transpose() - alpha * gradient).transpose()
            self.B = np.asarray(self.B).reshape(-1)

            # New Cost Value
            self.x_plot.append(iteration)
            self.y_plot.append(self.cost_function())

            print(iteration, self.y_plot[len(self.y_plot) - 1])

            if iteration > 1 and self.y_plot[iteration] - self.y_plot[iteration - 1] >= tol:
                break


    def normal_equation(self):
        """
        Implementação do normal equation
        """
        arr_id = np.identity(self.n)
        arr_id[0, 0] = 0
        lamda = 0
        self.B = pinv(self.X.T.dot(self.X) + lamda * arr_id).dot(self.X.T).dot(self.y)

