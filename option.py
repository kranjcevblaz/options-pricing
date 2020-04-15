import numpy as np
from option_calc import StockOption
import math


class Binomial(StockOption):
    """
    calculate required preliminary parameters:
    u = factor change of upstate
    d = factor change of downstate
    qu = risk free upstate probability
    qd = risk free downstate probability
    M = number of nodes
    """

    def __int_prm__(self):
        self.M = self.N + 1
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1 / self.u
        self.qu = (math.exp((self.r - self.div) * self.dt) - self.d) / (self.u - self.d)
        self.qd = 1 - self.qu

    def stocktree(self):
        stocktree = np.zeros([self.M, self.M])
        for i in range(self.M):
            for j in range(self.M):
                stocktree[j, i] = self.S0 * (self.u ** (i - j)) * (self.d ** j)
        return stocktree

    def option_price(self, stocktree):
        option = np.zeros([self.M, self.M])
        # the last node only: S0 - K
        # call: stock price - strike
        if self.is_call:
            option[:, self.M - 1] = np.maximum(np.zeros(self.M), (stocktree[:, self.N] - self.K))
        # put: strike - stock price
        else:
            option[:, self.M - 1] = np.maximum(np.zeros(self.M), (self.K - stocktree[:, self.N]))
        return option

    def option_value_tree(self, option, stocktree):
        # np.arange(start, stop, step) (-1) to include end 0 for sure
        for i in np.arange(self.M-2, -1, -1):
            for j in range(0, i+1):
                if self.american:
                    if self.is_call:
                        option[j, i] = np.maximum(stocktree[j, i] - self.K, math.exp(-self.r*self.dt) * (self.qu*option[j, i+1]+self.qd*option[j+1, i+1]))
                    else:
                        option[j, i] = np.maximum(self.K - stocktree[j, i], math.exp(-self.r*self.dt) * (self.qu*option[j, i+1]+self.qd*option[j+1, i+1]))
                else:
                    option[j, i] = math.exp(-self.r*self.dt) * (self.qu*option[j, i+1]+self.qd*option[j+1, i+1])
        return option

    def begin_tree(self):
        stocktree = self.stocktree()
        payoff = self.option_price(stocktree)
        return self.option_value_tree(payoff, stocktree)

    def price(self):
        self.__int_prm__()
        self.stocktree()
        payoff = self.begin_tree()
        # returns position 0,0 in array -> 1st cell in binomial tree
        return payoff[0, 0]
