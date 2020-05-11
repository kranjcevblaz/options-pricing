from option import Binomial
from black_scholes import BlackScholes

# choose inputs for Binomial or B-S methods

bin_option = Binomial(100, 102, 0.01, 0.5, 200,
                    {'tk': 'AAPL', 'is_calc': True, 'is_call': True, 'use_garch': True, 'start_date': '2018-01-01',
                     'end_date': '2018-06-28', 'american': False})

bs_option = BlackScholes(100, 102, 0.01, 0.5, 10,
                         {'tk': 'AAPL', 'is_calc': True, 'is_call': True, 'use_garch': True, 'start_date': '2018-01-01',
                          'end_date': '2018-06-28', 'american': False})

print(bin_option.price())
print(bs_option.black_scholes())
