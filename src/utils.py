import numpy as np
from py_vollib.black.implied_volatility import implied_volatility
from py_vollib.black import black
from py_vollib.black.greeks.analytical import vega

def dataset_split(x,y,test_size = 0.2):
    test_size = 0.2
    test_index = int(x.shape[0]*test_size)

    permutation = np.random.permutation(x.shape[0])
    permutation_test = permutation[:test_index]
    permutation_train = permutation[test_index:]

    x_train = x[permutation_train,:]
    y_train = y[permutation_train,:]
    x_test = x[permutation_test,:]
    y_test = y[permutation_test,:]

    return x_train, y_train, x_test, y_test

def price_to_volatility(F,T,K,market_prices,r = 0.0,q = "c"):
    ''' 
    F (float) – underlying futures price
    T (float) – time to expiration in years
    K (float) – strike price
    market_prices (float) – discounted Black price of a futures option
    r (float) – the risk-free interest rate
    q (str) – ‘p’ or ‘c’ for put or call
    '''

    bs_volatilities = np.zeros_like(market_prices)
    for i in range(market_prices.shape[0]):
        for j in range(market_prices.shape[1]):
            if market_prices[i,j]*np.exp(r*T[j])<F[j]-K[i,j]:
                bs_volatilities[i,j] = np.exp(r*T[j])*(F[j]-K[i,j])
            else:
                bs_volatilities[i,j] = implied_volatility(market_prices[i,j],F[j],K[i,j],r,T[j],q)
    return bs_volatilities

def volatility_to_price(F,T,K,market_volatilities,r = 0.0,q = "c"):
    '''
    F (float) – underlying futures price
    T (float) – time to expiration in years
    K (float) – strike price
    market_volatilities (float) – annualized standard deviation, or volatility
    '''
    prices = np.zeros_like(market_volatilities)
    for i in range(market_volatilities.shape[0]):
        for j in range(market_volatilities.shape[1]):
            prices[i,j] = black(q,F[j],K[i,j],T[j],r,market_volatilities[i,j])
    return prices

def volatility_to_vega(F,T,K,market_volatilities,r = 0.0,q = "c"):
    '''
    F (float) – underlying futures price
    T (float) – time to expiration in years
    K (float) – strike price
    market_volatilities (float) – annualized standard deviation, or volatility
    '''
    vegas = np.zeros_like(market_volatilities)
    for i in range(market_volatilities.shape[0]):
        for j in range(market_volatilities.shape[1]):
            vegas[i,j] = vega(q,F[j],K[i,j],T[j],r,market_volatilities[i,j])
    return vegas
