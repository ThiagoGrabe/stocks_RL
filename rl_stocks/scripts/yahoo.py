# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
import yfinance as yf


class Yahoo():
    """
    Implements Yahoo connection and methods to get the stocks data

    param period: date period or date window to download. Like (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    param stock:  stock ticket
    param start_date: the start date the user want to train
    param end_date: the final date the user want to train
    param interval: data interval. Like (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    yfinance: https://github.com/ranaroussi/yfinance
    """
    def __init__(self, window, stock, start_date, end_date, interval):
        self.PERIOD   = window
        self.STOCK    = stock
        self.START    = start_date
        self.END      = end_date
        self.INTERVAL = interval
        

    def getInfo(self):
        "Get stock information"
        self.data = yf.download(self.STOCK, start=self.START, end=self.END, interval=self.INTERVAL)
        return self.data

    def getStockInfo(self, key):
        "Given a key it returns a numpy array with the information"
        try:
            self.info = self.data[key].values
        except:
            self.getInfo()
            self.info = self.data[key].values
            # print('Could not access {} stock info'.format(info))

        return self.info

    def getStockState(self):
        "Return all stock information as a numpy array"
        try:
            return self.data.to_numpy()
        except:
            self.getInfo()
            return self.data.to_numpy()