from datetime import datetime, timedelta
import numpy as np
import scipy.io
import pandas as pd

from pandas.tseries.offsets import CustomBusinessDay
from pandas_market_calendars import get_calendar


class Vanilla:

    def __init__(self,start: str,p_maturity_futures: str,p_maturity_options: str,p_moneyness: str,p_volatilities: str,date_format: str = "%Y-%m-%d",calendar: str = "NYSE"):

        self.format = date_format
        self.calendar = get_calendar(calendar)
        self.offset = CustomBusinessDay(calendar = self.calendar.valid_days)
        self.start = datetime.strptime(start,self.format)

        self.maturity_futures =  np.loadtxt(p_maturity_futures,dtype = float,delimiter = ",")
        self.maturity_options = np.loadtxt(p_maturity_options,dtype = float,delimiter = ",")
        
        self.dates_futures = list(map(lambda x: (self.start+timedelta(days = int(x*365))+self.offset-self.offset).date(),self.maturity_futures))
        self.dates_options = list(map(lambda x: (self.start+timedelta(days = int(x*365))+self.offset-self.offset).date(),self.maturity_options))

        self.moneyness = np.loadtxt(p_moneyness,dtype = float,delimiter = ",")
        self.volatilities = np.loadtxt(p_volatilities,dtype = float,delimiter = ",")

        assert (self.moneyness.shape == self.volatilities.shape), "ERROR: shape of moneyness and volatilities don't match"
   
    @classmethod
    def from_folder(self,start: str,folder: str,date_format: str = "%Y-%m-%d",calendar: str = "NYSE"):
        p_maturity_futures = folder+"vanilla/maturity_futures.csv"
        p_maturity_options = folder+"vanilla/maturity_options.csv"
        p_moneyness = folder+"vanilla/moneyness.csv"
        p_volatilities = folder+"vanilla/market_volatilities.csv"
        return Vanilla(start,p_maturity_futures,p_maturity_options,p_moneyness,p_volatilities,date_format = date_format,calendar = calendar)
 
class Future:

    def __init__(self,start: str,p_dates: str,p_prices: str,date_format: str = "%Y-%m-%d",calendar: str = "NYSE"):

        self.format = date_format
        self.calendar = get_calendar(calendar)
        self.offset = CustomBusinessDay(calendar = self.calendar.valid_days)
        self.start = datetime.strptime(start,self.format)

        self.prices = np.loadtxt(p_prices,dtype = float,delimiter = ",")
        self.dates = np.loadtxt(p_dates,dtype = "str",delimiter = ",")
        self.dates = np.array(list(map(lambda x: datetime.strptime(x,self.format),self.dates)))
        

        assert (self.prices.shape == self.dates.shape), "ERROR: shape of moneyness and volatilities don't match"
   
    @classmethod
    def from_folder(self,start: str,folder: str,date_format: str = "%Y-%m-%d",calendar: str = "NYSE"):
        p_dates = folder+"futures/futures_dates.csv"
        p_prices = folder+"futures/futures_price.csv"
        return Future(start,p_dates,p_prices,date_format = date_format,calendar = calendar)

class Index:

    def __init__(self,start: str,p_dates: str,p_moneyness: str,p_volatilities: str,date_format: str = "%Y-%m-%d",calendar: str = "NYSE"):

        self.format = date_format
        self.calendar = get_calendar(calendar)
        self.offset = CustomBusinessDay(calendar = self.calendar.valid_days)
        self.start = datetime.strptime(start,self.format)

        self.dates = np.loadtxt(p_dates,dtype = "str",delimiter = ",")
        self.dates = np.array(list(map(lambda x: datetime.strptime(x,self.format),self.dates)))
        self.maturities = np.array(list(map(lambda x: (x-self.start).days/365.,self.dates)))

        self.moneyness = np.loadtxt(p_moneyness,dtype = float,delimiter = ",")
        self.volatilities = np.loadtxt(p_volatilities,dtype = float,delimiter = ",")

        assert (self.moneyness.shape == self.volatilities.shape), "ERROR: shape of moneyness and volatilities don't match"
   
    @classmethod
    def from_folder(self,start: str,folder: str,date_format: str = "%Y-%m-%d",calendar: str = "NYSE"):
        p_dates = folder+"index/index_dates.csv"
        p_moneyness = folder+"index/moneyness.csv"
        p_volatilities = folder+"index/volatilities.csv"
        return Index(start,p_dates,p_moneyness,p_volatilities,date_format = date_format,calendar = calendar)
    
    @staticmethod
    def from_mat(path):
        mat = scipy.io.loadmat(path)
        moneyness = mat["moneyness"]
        maturities = mat["month"][1:]
        vol_ask = mat["vol_ask"][:,2:-3,:]
        vol_bid = mat["vol_bid"][:,2:-3,:]
        date = pd.to_datetime(mat["date"].flatten()-719529, unit='D').strftime('%Y-%m-%d').values
        return moneyness, maturities, vol_ask, vol_bid, date


class Parameter:

    def __init__(self,filename: str):

        self.keys = ["a","rho","kappa","theta","chi","rhov","initial volatility"]
        self.parameters = np.loadtxt(filename,dtype = float,delimiter = ",")

   
    @classmethod
    def from_folder(self,folder: str):
        filename = folder+"parameters.csv"
        return Parameter(filename)
    
    @classmethod
    def from_folder(self,folder: str):
        filename = folder+"parameters.csv"
        return Parameter(filename)
