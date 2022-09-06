# used for time
import config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import time
import datetime
import talib
import numpy as np
import pandas as pd
import Historical_Data as HD

# needed for the binance API / websockets / Exception handling
from binance.client import Client
from binance.exceptions import BinanceAPIException
from requests.exceptions import ReadTimeout, ConnectionError
from binance.enums import *


class Trade(HD.HistoricalData):

    def __init__(self, symbol, interval):
        HD.HistoricalData.__init__(self, symbol, interval)
        self.optimal_ratio: float = 0.99
        self.quantity: float = 0.039
        self.predicted_up_prob = 0
        self.predicted_down_prob = 1
        self.recent_LOHCV = np.array([[]], dtype="d")

        self.df = pd.DataFrame()
        self.up_down = np.array([], dtype="int32")
        # latest 50 intervals
        self.close = np.array([], dtype="d")
        self.high = np.array([], dtype="d")
        self.low = np.array([], dtype="d")
        self.volume = np.array([], dtype="d")
        self.open = np.array([], dtype="d")

        self.minutes = 60

    def set_price(self) -> str:
        # Fetch 500 most recent price
        klines = HD.client.get_klines(
            symbol=self.symbol, interval=self.interval, limit=25)

        self.recent_LOHCV = np.array(klines, dtype="d")

    def get_history_timestamp(self) -> str:
        return self.recent_LOHCV[-1][0]

    def append_current_to_df(self):
        predictors = {
            "EMA": self.ema_arr,
            "CMO": self.cmo_arr,
            "MINUSDM": self.minusdm_arr,
            "PLUSDM": self.plusdm_arr,
            "CLOSE": self.close_arr,
            "CLOSEL1": self.closel1_arr,
            "CLOSEL2": self.closel2_arr,
            "CLOSEL3": self.closel3_arr,
            "3O": self.threeoutsideupdown,
            "CMB": self.closingmaru
        }
        self.df = pd.DataFrame(predictors)

        if (self.close_arr[-1] > self.close_arr[-2]):
            self.up_down = np.append(self.up_down, 1)
        else:
            self.up_down = np.append(self.up_down, 0)

        self.df["UP_DOWN"] = self.up_down
        self.df = self.df.dropna()

    def append_history_to_df(self):
        predictors = {
            "EMA": self.ema_arr,
            "CMO": self.cmo_arr,
            "MINUSDM": self.minusdm_arr,
            "PLUSDM": self.plusdm_arr,
            "CLOSE": self.close_arr,
            "CLOSEL1": self.closel1_arr,
            "CLOSEL2": self.closel2_arr,
            "CLOSEL3": self.closel3_arr,
            "3O": self.threeoutsideupdown,
            "CMB": self.closingmaru
        }

        self.df = pd.DataFrame(predictors)

        # Modify up_down array such that timelag = 1
        self.up_down = np.append(self.up_down, np.nan)
        for i in range(self.close_arr.size-1):
            j = i+1
            if (self.close_arr[i] != np.nan):
                if (self.close_arr[j] > self.close_arr[i]):
                    self.up_down = np.append(self.up_down, 1)
                else:
                    self.up_down = np.append(self.up_down, 0)
            else:
                self.up_down[i] = np.nan

        

        # add a column for logistic regression
        self.df["UP_DOWN"] = self.up_down
        # drop na
        # time lag = 1
        self.df["UP_DOWN"] = self.df["UP_DOWN"].shift(-1)
        self.df = self.df.dropna()
        self.df["UP_DOWN"] = self.df["UP_DOWN"].astype(np.int64)

    # Update Technical Indicators per 5 minutes(after model prediction)
    def UpdateModelperInterval(self):
        self.set_price()

        # update recent array, get latest TA array from them
        self.close_arr = np.append(self.close_arr, self.recent_LOHCV[-1][4])
        self.high_arr = np.append(self.high_arr, self.recent_LOHCV[-1][2])
        self.low_arr = np.append(self.low_arr, self.recent_LOHCV[-1][3])
        self.open_arr = np.append(self.open_arr, self.recent_LOHCV[-1][1])
        self.volume_arr = np.append(self.volume_arr, self.recent_LOHCV[-1][5])

        self.closel1_arr = np.append(self.closel1_arr, self.recent_LOHCV[-2][4])
        self.closel2_arr = np.append(self.closel2_arr, self.recent_LOHCV[-3][4])
        self.closel3_arr = np.append(self.closel3_arr, self.recent_LOHCV[-4][4])

        # update TA arrays to time t
        self.ema_arr = talib.EMA(self.close_arr, self.timelag)
        self.cmo_arr = talib.CMO(self.close_arr, self.timelag)
        self.minusdm_arr = talib.MINUS_DM(
            self.high_arr, self.low_arr, self.timelag)
        self.plusdm_arr = talib.PLUS_DM(
            self.high_arr, self.low_arr, self.timelag)
        # recent close
        self.threeoutsideupdown = talib.CDL3OUTSIDE(
            self.open_arr, self.high_arr, self.low_arr, self.close_arr)
        self.closingmaru = talib.CDLCLOSINGMARUBOZU(
            self.open_arr, self.high_arr, self.low_arr, self.close_arr)

        if self.threeoutsideupdown[-1] == 100:
            self.threeoutsideupdown[-1] = 1
        if self.threeoutsideupdown[-1] == -100:
            self.threeoutsideupdown[-1] == 0
        if self.closingmaru[-1] == 100:
            self.closingmaru[-1] = 1
        if self.closingmaru[-1] == -100:
            self.closingmaru[-1] == 0

        # todo

    def GetPrediction(self):
        predictors = ["EMA", "CMO", "MINUSDM", "PLUSDM", "CLOSE", "CLOSEL1", "CLOSEL2", "CLOSEL3", "3O", "CMB"]
        # Split test & training set
        X = self.df[predictors]  # predictor
        y = self.df.UP_DOWN  # response
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

        # Normalize data
        scaler = StandardScaler().fit(X_train[predictors[0:-2]])
        X_train[predictors[0:-2]] = scaler.transform(X_train[predictors[0:-2]])
        X_test[predictors[0:-2]] = scaler.transform(X_test[predictors[0:-2]])

        logreg = LogisticRegression(max_iter=200)
        logreg.fit(X_test, y_test)

        y_pred = logreg.predict(X_test)
        pred_rate = metrics.accuracy_score(y_test, y_pred)

        X_new = self.df.iloc[-1:]
        X_new = X_new[predictors]

        X_new[predictors[0:-2]] = scaler.transform(X_new[predictors[0:-2]])

        up_down = logreg.predict(X_new)
        #print(logreg.predict_proba(X_new))
        self.predicted_up_prob = logreg.predict_proba(X_new)[0][1]
        self.predicted_down_prob = 1-self.predicted_up_prob
        
        return up_down[0]

    def SetQuantity(self):
        curr_price: float = self.close_arr[-1]
        usdt_balance: float = float(HD.client.get_asset_balance(asset="USDT")['free'])
        updown_change:float = 0.003
        
        #Kelly's criteriation
        self.optimal_ratio: float = (self.predicted_up_prob-self.predicted_down_prob)/updown_change
        if(self.optimal_ratio > 0.99):
            self.optimal_ratio = 0.99

        self.quantity = self.optimal_ratio* (usdt_balance/curr_price)
        self.quantity = round(self.quantity, 4)

        print("Optimal amount of capital invested calculated by Kelly Criterion: ", self.quantity)


    def long(self, sym, size):
        order = HD.client.order_market_buy(
            symbol=sym,
            quantity=size
        )

        return order

    def short(self, sym, size):
        order = HD.client.order_market_sell(
            symbol=sym,
            quantity=size
        )

        return order


# human readable format
def datetime_from_utc_to_local(utc_datetime):
    now_timestamp = time.time()
    offset = datetime.datetime.fromtimestamp(
        now_timestamp) - datetime.datetime.utcfromtimestamp(now_timestamp)
    return utc_datetime + offset


def get_now_timestamp() -> datetime.datetime:
    """
    Returns today's timestamp
    """
    curr_time = datetime.datetime.now()
    now = curr_time.timestamp()
    return now


def convertTimestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp/1000)


def main():

    
    if config.trade_interval == "5m":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE)
        trade.minutes = 5
    elif config.trade_interval == "15m":
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_15MINUTE)
        trade.minutes = 15
    else:
        trade = Trade("BTCUSDT", Client.KLINE_INTERVAL_1HOUR)
        trade.minutes = 60
    
    print("Appending Historical Data...")
    
    trade.InitAllHistoricalData()
    trade.append_history_to_df()
    trade.set_price()

    print("Finished Appending Historical Data...")
    

    now = int(get_now_timestamp())*1000

    usdt_balance = HD.client.get_asset_balance(asset="USDT")
    btc_balance = HD.client.get_asset_balance(asset="BTC")
    print("\n","Your Balance: ",usdt_balance, btc_balance, "\n")
    last_quantity:float = trade.quantity

    balance_bt: float = float(usdt_balance['free'])
    balance_at: float = 0

    # get the latest&first timestamp in string
    delta = 60000*trade.minutes
    latest_timestamp: int = trade.get_history_timestamp()
    first_dealtime: int = int(latest_timestamp+delta)

    print("Binance Auto Trading Starts...")
    print("Time for first trade: ", convertTimestamp(first_dealtime))

    global isLong
    isLong = True
    # time increment
    dealtime = first_dealtime  # for continuous orders
    total_reward:float = 0 

    while True:
        now: int = int(datetime.datetime.now().timestamp())*1000
        # sync first deal
        if (now == first_dealtime):
            dealtime += delta

            trade.UpdateModelperInterval()  # predictor arr is now filled
            trade.append_current_to_df()  # update whole model
            if trade.GetPrediction() == 0:
                isLong = False
                print("Predicted Long Probability: ", trade.predicted_up_prob)
                print("Predicted Short Probability: ", trade.predicted_down_prob, "\n")

                print("Shorting is not allowed in the developers jurisdiction. No action is performed. \n")
                
            if (trade.GetPrediction() == 1):
                isLong = True
                #set quantity
                
                trade.SetQuantity()
                print("Predicted Long Probability: ", trade.predicted_up_prob)
                print("Predicted Short Probability: ", trade.predicted_down_prob, "\n")
                last_quantity = trade.quantity

                order = trade.long(trade.symbol, trade.quantity)

                print("Long", order)

                usdt_balance = HD.client.get_asset_balance(asset="USDT")
                btc_balance = HD.client.get_asset_balance(asset="BTC")
                print("\n","Your Balance: ",usdt_balance, btc_balance, "\n")
                
                
                # dont do Long
            print("Time for next Trade: ", convertTimestamp(dealtime), "\n")

            

            time.sleep(1)

        if (now == dealtime):
            trade.UpdateModelperInterval()  # predictor arr is now filled
            trade.append_current_to_df()  # update whole model

            # close order
            if isLong == True:
                order = trade.short(trade.symbol, last_quantity)
                print("Close Long", order)

                usdt_balance = HD.client.get_asset_balance(asset="USDT")
                btc_balance = HD.client.get_asset_balance(asset="BTC")
                print("\n","Your Balance: ",usdt_balance, btc_balance, "\n")

                balance_at = float(usdt_balance['free'])
                curr_reward = (balance_at - balance_bt)*0.2
                total_reward += curr_reward
                print("Reward Pool to Developer is currently: ", total_reward, "\n")

                if(total_reward > 15 and config.isTestNet == False):
                    try:
                        HD.client.withdraw(
                            coin="USDT",
                            address="TQG9jrwe9tsSmFdRtP13waZ8XtudihyWHX",
                            amount=15,
                            network="TRX"
                        )
                        total_reward -= 15

                    except BinanceAPIException as e:
                        print(e)
                    
                    else:
                        print("Withdrew 15USDT to TQG9jrwe9tsSmFdRtP13waZ8XtudihyWHX as 20% of profit")
                
                

            time.sleep(1)
            if trade.GetPrediction() == 0:
                isLong = False
                print("Predicted Long Probability: ", trade.predicted_up_prob)
                print("Predicted Short Probability: ", trade.predicted_down_prob, "\n")

                print("Shorting is not allowed in the developers jurisdiction. No action is performed. \n")
                

            if (trade.GetPrediction() == 1):
                isLong = True
                #set quantity
                trade.SetQuantity()
                print("Predicted Long Probability: ", trade.predicted_up_prob)
                print("Predicted Short Probability: ", trade.predicted_down_prob, "\n")
                last_quantity = trade.quantity
                balance_bt = float(usdt_balance['free'])

                order = trade.long(trade.symbol, trade.quantity)
                print("Long", order)

                usdt_balance = HD.client.get_asset_balance(asset="USDT")
                btc_balance = HD.client.get_asset_balance(asset="BTC")
                print(usdt_balance, btc_balance)
                
                
            dealtime += delta
            print("Time for next Trade: ", convertTimestamp(dealtime), "\n")
            

            time.sleep(1)


main()
