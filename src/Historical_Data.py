from binance import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException

from time import sleep
from sys import exit
import config

import numpy as np
import pandas as pd
import talib

client = Client(config.api_key, config.api_secret, testnet=config.isTestnet)
client_query = Client(config.api_key, config.api_secret)

#Change TA functions to setters instead of getters
class HistoricalData:

    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.arr = np.array([], dtype='d')
        self.timelag = 20

        self.close_arr = np.array([], dtype="d")
        self.high_arr = np.array([], dtype="d")
        self.low_arr = np.array([], dtype="d")
        self.volume_arr = np.array([], dtype="d")
        self.open_arr = np.array([], dtype="d")

        self.closel1_arr = np.array([], dtype="d")
        self.closel2_arr = np.array([], dtype="d")
        self.closel3_arr = np.array([], dtype="d")

        #Momentum indicators
        self.ema_arr = np.array([], dtype="d")
        self.cmo_arr = np.array([], dtype="d")
        self.minusdm_arr = np.array([], dtype="d")
        self.plusdm_arr = np.array([], dtype="d")


        #Pattern recognition

        self.threeoutsideupdown = np.array([], dtype="int")
        self.closingmaru = np.array([], dtype="int")



    def SetKLineData(self):
        test = client_query.get_historical_klines(symbol=self.symbol, interval=self.interval ,start_str="1 Jul, 2020", end_str="4 Sep, 2022")
        self.arr = np.array(test, dtype="d")
        
    def InitAllHistoricalData(self):
        self.SetKLineData()

        for i in self.arr:
            self.open_arr = np.append(self.open_arr, i[1])
            self.close_arr = np.append(self.close_arr, i[4])
            self.high_arr = np.append(self.high_arr, i[2])
            self.low_arr = np.append(self.low_arr, i[3])
            self.volume_arr = np.append(self.volume_arr, i[5])
        
        # Creating timelag arrays for closing price
        for i in range(1, self.close_arr.size):
            self.closel1_arr = np.append(self.closel1_arr, self.close_arr[i])
        
        for i in range(2, self.close_arr.size):
            self.closel2_arr = np.append(self.closel2_arr, self.close_arr[i])

        for i in range(3, self.close_arr.size):
            self.closel3_arr = np.append(self.closel3_arr, self.close_arr[i])
        
        self.closel1_arr = np.append(self.closel1_arr, [np.nan])
        self.closel2_arr = np.append(self.closel2_arr, [np.nan, np.nan])
        self.closel3_arr = np.append(self.closel3_arr, [np.nan, np.nan, np.nan])
        

        self.ema_arr = talib.EMA(self.close_arr, self.timelag)
        self.minusdm_arr = talib.MINUS_DM(self.high_arr, self.low_arr, self.timelag)
        self.plusdm_arr = talib.PLUS_DM(self.high_arr, self.low_arr, self.timelag)
        self.cmo_arr = talib.CMO(self.close_arr, self.timelag)


        #patterns
        self.threeoutsideupdown = talib.CDL3OUTSIDE(self.open_arr, self.high_arr, self.low_arr, self.close_arr)
        self.closingmaru = talib.CDLCLOSINGMARUBOZU(self.open_arr, self.high_arr, self.low_arr, self.close_arr)
        
        for i in range(self.closingmaru.size):
            if self.closingmaru[i] == 100:
                self.closingmaru[i] = 1
            if self.closingmaru[i] == -100:
                self.closingmaru[i] = 1

        for i in range(self.threeoutsideupdown.size):
            if self.threeoutsideupdown[i] == 100:
                self.threeoutsideupdown[i] = 1
            if self.threeoutsideupdown[i] == -100:
                self.threeoutsideupdown[i] = 0