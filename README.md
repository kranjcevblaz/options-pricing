# Binomial and Black-Scholes Option Pricing 

A basic model to price vanilla European and American options. The binomial model can be used for both American and European options
but Black-Scholes can only be used to price European ones. The model also includes calculation of implied volatility for B-S model.
Binomial model is easier to understand and more intuitive, this allows you to tweak the model for other derivatives or add extra parameters.
B-S method should provide higher accuracy for European options but it's often harder to understand and more of a black-box solution.

## Supporting libraries
The core functionality depends on Python's ```pandas, numpy and scipy```. The stock volatility data is sourced from Alpha Vantage API. To implement it, you will
need an API key. Please specify it in ```stock_volatility.py``` in ```<ALPHA-VANTAGE-KEY>``` placeholder. Volatility can be estimated via simple mean sigma calculation
or more advanced general ARCH model provided by ARCH library. Install the packages with ```pip```:

```pip install pandas, numpy, scipy, alpha-vantage, arch```

https://machinelearningmastery.com/develop-arch-and-garch-models-for-time-series-forecasting-in-python/

## Instructions
To use the model, specify the parameters in a dictionary within a specific class in ```run_pricing.py```. Enter the parameters as such:

```python
bin_option = Binomial('stock price', 'strike', 'risk-free rate', 'time-step', 'no. of steps', '{dict of extra parameters}')
```

## Future updates
I am planning to implement pricing for exotic options with trinomial tree model. This will also provide higher accuracy for vanilla option.
I am also working on a front-end code to visualise the results and provide easier input. 

Feel free to clone the repo and expand it to your needs.
