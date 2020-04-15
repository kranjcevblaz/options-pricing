from stock_volatility import StockVol
import math


class StockOption:

    def __init__(self, S0, K, r, T, N, prm):

        """
        S0: initial stock price
        K: strike price
        r: risk free rate
        T: length of the option in years
        N: no. of steps in the tree
        prm: dictionary with additional params
        type S0: object
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.N = N

        '''
        prm parameters:
        start = date from when you want to analyse stocks, "yyyy-mm-dd"
        end = date of final stock analysis (likely current date), "yyyy-mm-dd"
        tk = ticker label
        div = dividend paid
        is_calc = is volatility calculated using stock price history, boolean
        use_garch = use GARCH model, boolean
        sigma = volatility of stock
        is_call = is it a call option, boolean
        eu_option = European or American option, boolean
        '''

        self.tk = prm.get('tk', None)
        self.start = prm.get('start', None)
        self.end = prm.get('end', None)
        self.div = prm.get('div', 0)
        self.is_calc = prm.get('is_calc', False)
        self.use_garch = prm.get('use_garch', False)
        self.vol = StockVol(self.tk, self.start, self.end)

        if self.is_calc:
            if self.use_garch:
                self.sigma = self.vol.garch_sigma()
            else:
                self.sigma = self.vol.mean_sigma()
        # if no sigma is passed from mean or GARCH model, use manually entered one, otherwise 0
        else:
            self.sigma = prm.get('sigma', 0)

        self.is_call = prm.get('is_call', True)
        self.american = prm.get('american', True)
        '''
        derived values:
        dt = time per step, in years
        df = discount factor
        '''
        self.dt = T / float(N)
        self.df = math.exp(-(r - self.div) * self.dt)
