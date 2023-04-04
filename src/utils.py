import numpy as np
from py_vollib.black.implied_volatility import implied_volatility
from py_vollib.black import black
from py_vollib.black.greeks.analytical import vega

#def dataset_split(x,y,test_size = 0.2):
#    test_size = 0.2
#    test_index = int(x.shape[0]*test_size)
#
#    permutation = np.random.permutation(x.shape[0])
#    permutation_test = permutation[:test_index]
#    permutation_train = permutation[test_index:]
#
#    x_train = x[permutation_train,:]
#    y_train = y[permutation_train,:]
#    x_test = x[permutation_test,:]
#    y_test = y[permutation_test,:]
#
#    return x_train, y_train, x_test, y_test

def image_to_dat(array,fname: str):
    x = np.tile(np.arange(array.shape[1]),array.shape[0])[:,None]
    y = np.repeat(np.arange(array.shape[0]),array.shape[1])[:,None]
    z = array.flatten()[:,None]
    np.savetxt(fname,np.concatenate((x,y,z),axis = 1),
        delimiter = " ",header = "x y z",comments = "")

def dataset_split(data,test_size = 0.2,debug = True):
    if debug:
        np.random.seed(0)

    test_size = 0.2
    size = len(data[0])
    test_index = int(size*test_size)

    permutation = np.random.permutation(size)
    permutation_test = permutation[:test_index]
    permutation_train = permutation[test_index:]
    
    train = []
    test = []
    for i in range(len(data)):
        train.append(data[i][permutation_train,:])
        test.append(data[i][permutation_test,:])
        

    return tuple(train), tuple(test)

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

def marginal(data, values, aggregation_method='mean'):
    """
    Computes the aggregated values of a dataset for each unique data value.

    Args:
        data (array-like): The dataset to aggregate.
        values (array-like): The values associated with each data point.
        aggregation_method (str or function, optional): The method to use for aggregating the values.
            If a string, it should be one of 'mean', 'median', 'sum', 'min', or 'max'.
            If a function, it should take an array of values as input and return a single aggregated value.

    Returns:
        Two arrays: one with the unique data values and one with the aggregated values for each unique data value.
    """
    # create a dictionary to hold the values for each data point
    data_dict = {}

    # loop through the data and values arrays and add up the values for each data point
    for i in range(len(data)):
        key = data[i]
        value = values[i]
        if key not in data_dict:
            data_dict[key] = [value]
        else:
            data_dict[key].append(value)

    # aggregate the values for each data point
    if isinstance(aggregation_method, str):
        if aggregation_method == 'mean':
            aggregation_function = np.mean
        elif aggregation_method == 'median':
            aggregation_function = np.median
        elif aggregation_method == 'sum':
            aggregation_function = np.sum
        elif aggregation_method == 'min':
            aggregation_function = np.min
        elif aggregation_method == 'max':
            aggregation_function = np.max
        else:
            raise ValueError("Invalid aggregation method specified.")
    elif callable(aggregation_method):
        aggregation_function = aggregation_method
    else:
        raise TypeError("Aggregation method must be a string or function.")

    data_array = []
    aggregated_values_array = []
    for key in data_dict:
        aggregated_value = aggregation_function(data_dict[key])
        data_array.append(key)
        aggregated_values_array.append(aggregated_value)

    return np.array(data_array), np.array(aggregated_values_array)
