from alpha_vantage.timeseries import TimeSeries
# from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# from statsmodels.graphics.tsaplots import plot_acf
import arch

ts = TimeSeries(key='4F9ZORTL1064280X', output_format='pandas')


class StockVol:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        data, meta_data = ts.get_daily_adjusted(symbol=self.ticker, outputsize='full')
        self.stock_data = pd.DataFrame(data[self.end_date:self.start_date]['4. close'])
        # np.log - difference in log values between 1 to next day
        self.stock_data['log'] = np.log(self.stock_data) - np.log(self.stock_data.shift(1))

    # can use either of them
    # regular mean volatility(sigma)
    # @property
    def mean_sigma(self):
        st = self.stock_data['log'].dropna().ewm(span=252).std()
        sigma = st.iloc[-1]
        return sigma

    # general  ARCH volatility (sigma)
    # @property
    def garch_sigma(self):
        model = arch.arch_model(self.stock_data['log'].dropna(), mean='Zero', vol='GARCH', p=1, q=1, rescale=False)
        model_fit = model.fit()
        forecast = model_fit.forecast(horizon=1)
        var = forecast.variance.iloc[-1]
        sigma = float(np.sqrt(var))
        # print(sigma)
        return sigma


if __name__ == "__main__":
    vol = StockVol('AAPL', start_date='2017-08-18', end_date='2018-08-18')
    print(vol.garch_sigma())
    # test = vol.stock_data['log'].dropna()
    # garch_sigma()
    # fig = plot_acf(test)
    # plt.show()
