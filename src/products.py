from datetime import datetime, timedelta
import numpy as np
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
 
