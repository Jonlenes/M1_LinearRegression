from time import time
import numpy as np
from plot_cost_function import PlotCostFunction


class LR:
    """
    Implementação básica de regressão linear - com programação dinamica
    """

    thetas, X, y = [], [], []
    m, n = 0, 0

    x_plot, y_plot = [], []
    conv_point = None
    pd = None

    def __init__(self, alphas=[0.006]):
        self.alphas = alphas


    def fit(self, x, y):
        """
        Realiza o treinamento do modelo

        Parameters
        ----------
        x : array_like
            features.
        y : array_like
            targets
        """

        self.X = x
        self.y = y

        self.X = np.append(np.ones((len(self.X), 1)), self.X, axis=1)

        self.m = self.X.shape[0]
        self.n = self.X.shape[1]

        plot = PlotCostFunction(self.alphas)

        for alpha in self.alphas:
            self.gradient_descent(alpha, 1000)
            plot.add_function(self.x_plot, self.y_plot)
            
        plot.show()


    def _hypothesis(self, x):
        """
        Realiza a predição para um exemplo x

        Parameters
        ----------
        x : array_like
            features de um exemplo.

        Returns
        -------
        out : ndarray
            Valor interpolado para cada exemplo de x
        
        """
        return np.sum(np.multiply(x, self.thetas))


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
        y = []
        for x in x_test:
            y.append(self._hypothesis(x))
        return y


    def _compute_diff(self, i_thetas, square_sum=False):
        """
        Método que computa as diferenças (real vs predict) com pd
        """
        if self.pd is None and not square_sum:
            self.pd = [0] * self.m
            for i in range(0, self.m):
                self.pd[i] = self._hypothesis(self.X[i]) - self.y[i]

        sum = 0
        if square_sum:
            self.pd = [0] * self.m
            for i in range(0, self.m):
                self.pd[i] = self._hypothesis(self.X[i]) - self.y[i]
                sum += (self.pd[i] * self.pd[i])
        else:
            for i in range(0, self.m):
                sum += self.pd[i] * self.X[i, i_thetas]

        return sum


    def cost_function(self):
        """
        Calcula o valor da função de custo de todos armazenados no treinamento 

        Returns
        -------
        out : long
           valor do custo
        """
        return (1. / (2. * self.m)) * self._compute_diff(-1, True)


    def gradient_descent(self, alpha=.1, max_iter=5, tol=.00001):
        """
        Implementação do gradient descent
        """
        count_iter = 0
        self.x_plot = []
        self.y_plot = []
        self.thetas = np.zeros(self.n)
        self.pd = None

        while True:
            thetas_temp = np.zeros(self.n)  # thetas temporarios

            for i_thetas in range(0, self.n):
                thetas_temp[i_thetas] = self.thetas[i_thetas] - alpha * (1. / self.m) * self._compute_diff(i_thetas)

            self.thetas = thetas_temp
            self.pd = None
            count_iter += 1

            self.x_plot.append(count_iter)
            self.y_plot.append(self.cost_function())

            if self.conv_point is None and count_iter > 1 and self.y_plot[count_iter - 1] > self.y_plot[count_iter - 2]:
                self.conv_point = (count_iter, self.y_plot[count_iter - 2])

            print(count_iter, self.y_plot[len(self.y_plot) - 1])

            if count_iter >= max_iter: break
            if count_iter > 1 and abs(self.y_plot[count_iter - 1] - self.y_plot[count_iter - 2]) < tol: break
            if count_iter >= 5 and self.y_plot[count_iter - 1] > self.y_plot[count_iter - 2]: break
