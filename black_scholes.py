import numpy as np
from scipy import stats
import scipy.stats as si
from sympy.stats import Normal, cdf
from option_calc import StockOption


class BlackScholes(StockOption):
    def __init__(self, S0, K, r, T, N, prm, option_price):  # option_price - arg only used to calc implied volatility
        super().__init__(S0, K, r, T, N, prm)
        self.n = Normal('x', 0.0, 1.0)
        self.d1 = (np.log(self.S0 / self.K) + (self.r - 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)
        self.option_price = option_price

        if self.american:
            raise ValueError(
                'Black - Scholes only used for European options, use binomials for American option instead')

    def black_scholes(self):

        if self.is_call:
            price = np.exp(-self.r * self.T) * (self.S0 * stats.norm.cdf(self.d1) - self.K * stats.norm.cdf(self.d2))
        else:
            price = np.exp(-self.r * self.T) * (self.K * stats.norm.cdf(-self.d2) - self.S0 * stats.norm.cdf(-self.d1))

        return price

    def implied_vol(self):  # Newton's method for root finding with iterations

        if self.is_call:
            fx = self.S0 * si.norm.cdf(self.d1, 0.0, 1.0) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2,
                                                                                                            0.0,
                                                                                                            1.0) - self.option_price
        else:
            fx = self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2, 0.0, 1.0) - self.S0 * si.norm.cdf(-self.d1,
                                                                                                             0.0,
                                                                                                             1.0) - self.option_price

        vega = (1 / np.sqrt(2 * np.pi)) * self.S0 * np.sqrt(self.T) * np.exp(-(si.norm.cdf(self.d1, 0.0, 1.0) ** 2) * 0.5)

        tolerance = 1e-3
        x0 = self.sigma
        x_new = x0
        x_old = x0 - 1

        while abs(x_new - x_old) > tolerance:
            x_old = x_new
            x_new = (x_new - fx - self.option_price) / vega

            return abs(x_new)


if __name__ == '__main__':
    test = BlackScholes(25, 20, 0.05, 1, 10,
                        {'tk': 'AAPL', 'is_call': True, 'sigma': 0.25, 'use_garch': True, 'start_date': '2017-08-18',
                         'end_date': '2018-08-18', 'american': False}, 7)
    print(test.implied_vol())
