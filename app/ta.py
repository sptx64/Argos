import pandas as pd
import streamlit as st
import numpy as np
from scipy.signal import argrelextrema



def df_check() :
    df = pd.DataFrame({'ta_ref' : ['touching_MA','above_rsi', 'under_rsi', 'above_bb', 'under_bb', 'hu_mean', 'um', 'ham', 'squeeze', 'tweezer','divergence', 'compression', 'volume', "wick"], 'result' : ['', '','','','','','','','','','','','','']})
    #df = pd.DataFrame({'ta_ref' : ['touching_MA','above_rsi', 'under_rsi', 'above_bb', 'under_bb', 'hu_mean', 'um', 'ham', 'squeeze', 'tweezer','divergence'], 'result' : ['','','','','','','','','','','']})
    return df

def MA(df, window) :
    sma=df['Close'].rolling(window).mean()
    return sma

def touching_MA(df, window):
    df['MA'] = MA(df, window)
    last_row = df.iloc[-1]
    state = False
    if (last_row['MA'] >= last_row['Low']) & (last_row['MA'] <= last_row['High']) :
        state = True
    return state

def STD(df, window) :
    std = df['Close'].rolling(window).std()
    return std


def EMA(df, window) :
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return ema


def RSI(df, periods):
    close_delta = df['Close'].diff()
    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    rsi = ma_up / ma_down
    rsi = 100 - (100/(1 + rsi))
    return rsi


def check_if_above_rsi(df, periods, ab_rsi_number) :
    df['RSI'] = RSI(df, periods)
    state = False
    if df.iloc[-1]['RSI'] >= ab_rsi_number :
        state = True
    return state

def check_if_under_rsi(df, periods, un_rsi_number) :
    df['RSI'] = RSI(df, periods)
    state = False
    if df.iloc[-1]['RSI'] <= un_rsi_number :
        state = True
    return state


def HU(df) :
    df1 = df
    df1['umb_ham'] = 'none'
    if len(df)>0 :
        df1 = df
        df1['body'] = df1['High'] - df1['Low']
        df1.loc[(df1['Open'] >= (df1['Low'] + df1['body']*0.64)) & (df1['Close'] >= (df1['Low'] + df1['body']*0.64)), 'umb_ham'] = 'umbrella'
        df1.loc[(df1['Open'] <= (df1['Low'] + df1['body']*0.36)) & (df1['Close'] <= (df1['Low'] + df1['body']*0.36)), 'umb_ham'] = 'hammer'
    return df1['umb_ham']

def check_H(df) :
    hu = HU(df)
    state = False
    if len(hu) > 2 :
        if (hu.iloc[-1] == 'hammer') or (hu.iloc[-2] == 'hammer') :
            state = True
    return state

def check_U(df) :
    hu = HU(df)
    state = False
    if len(hu) > 2 :
        if (hu.iloc[-1] == 'umbrella') or (hu.iloc[-2] == 'umbrella') :
            state = True
    return state

def get_HU_mean(df, window) :
    df1 = df
    df1['umb_ham'] = HU(df1)
    df1['score_hu'] = 0
    df1['count_hu'] = 0

    #highest of open close
    df1['highest_oc'] = df1['Open']
    df1.loc[df1['Open'] < df1['Close'], 'highest_oc'] = df1['Close']

    #lowest of open close
    df1['lowest_oc'] = df1['Open']
    df1.loc[df1['Open'] > df1['Close'], 'lowest_oc'] = df1['Close']

    df1.loc[df1['umb_ham'] == 'umbrella', 'count_hu'] = 1
    df1.loc[df1['umb_ham'] == 'hammer', 'count_hu'] = -1
    count_hu = df1['count_hu'].rolling(window).sum()

    df1.loc[df1['umb_ham'] == 'umbrella', 'score_hu'] = 1 * (df['lowest_oc'] - df['Low'])
    df1.loc[df1['umb_ham'] == 'hammer', 'score_hu'] = -1 * (df['High'] - df['highest_oc'])
    mean_hu = df1['score_hu'].rolling(window).sum()
    return mean_hu, count_hu

def check_HU_mean(df, window) :
    mean_hu, count_hu = get_HU_mean(df, window)
    val = 1/10
    state = str(0)
    if (mean_hu.iloc[-1] >= (df['Close'].iloc[-1])*val) & (count_hu.iloc[-1] >= 5) :
        emoji = ' 🐸'
        if (count_hu.iloc[-1] >= 8) :
            emoji = ' 🐮'
        state = str(mean_hu.iloc[-1]) + '_' + str(count_hu.iloc[-1]) + emoji
    elif (mean_hu.iloc[-1] <= (-df['Close'].iloc[-1])*val) & (count_hu.iloc[-1] <= -5) :
        emoji = ' 🐻'
        if (count_hu.iloc[-1] <= -8) :
            emoji = ' 🐻‍❄️'
        state = str(mean_hu.iloc[-1]) + '_' + str(count_hu.iloc[-1]) + emoji
    return state


def get_bollinger_bands(df, window):
    sma = MA(df, window)
    std = STD(df, window)
    bb_up = sma + std * 2  # Calculate top band
    bb_low = sma - std * 2  # Calculate bottom band
    return bb_up, bb_low

def check_above_bband(df, window) :
    df['bb_up'], df['bb_down'] = get_bollinger_bands(df, window)
    last_row = df.iloc[-1]
    state = False
    if last_row['High'] >= last_row['bb_up'] :
        state = True
    return state

def check_under_bband(df, window) :
    df['bb_up'], df['bb_down'] = get_bollinger_bands(df, window)
    last_row = df.iloc[-1]
    state = False
    if last_row['Low'] <= last_row['bb_down'] :
        state = True
    return state

def get_kc(df, window, multiplier, atr_lookback): # 20, 2, 10
    hl = df['High'] - df['Low']
    ahc = abs(df['High'] - df['Close'].shift())
    alc = abs(df['Low'] - df['Close'].shift())
    frames = [hl, ahc, alc]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(alpha = 1/atr_lookback).mean()
    kc_middle = EMA(df, window)
    kc_upper = EMA(df, window) + multiplier * atr
    kc_lower = EMA(df, window) - multiplier * atr

    return kc_middle, kc_upper, kc_lower


def get_squeeze(df, window) :
    df1 = df
    bb_up, bb_low = get_bollinger_bands(df, window)
    kc_middle, kc_upper, kc_lower = get_kc(df, window, 2, window/2)
    df1['squeeze'] = 'off'
    df1.loc[(bb_up <= kc_upper) & (bb_low >= kc_lower), 'squeeze'] = 'on 🧨'
    df1.loc[(df1['squeeze'] == 'off') & (df1['squeeze'].shift(1) == 'on 🧨'), 'squeeze'] = 'released 💥'
    return df1['squeeze']

def check_if_sqz(df, window) :
    col_sqz = get_squeeze(df, window)
    state = 'off'
    if (col_sqz.iloc[-1] == 'on 🧨') | (col_sqz.iloc[-1] == 'released 💥') :
        state = col_sqz.iloc[-1]
    return state

def get_tweezer(df, range_accept = 0.1/100, percent_retrace = 0.33/100) :
    bb_up, bb_low = get_bollinger_bands(df, 20)
    ma20, ma50 = MA(df, 20), MA(df, 50)
    close = df['Close']
    openn = df['Open']

    upper_accept = close.shift(1) + (close.shift(1)*range_accept)
    lower_accept = close.shift(1) -  (close.shift(1)*range_accept)

    cond1 = (openn > lower_accept) & (openn < upper_accept)
    #Bullish scenario // Bottom call
    cond2_bull = (openn <= close) & (openn.shift(1) >= close.shift(1))
    #Bearish scenario // Top call
    cond2_bear = (openn >= close) & (openn.shift(1) <= close.shift(1))

    cond3_bull = (close >= (close.shift(1) + (openn.shift(1) - close.shift(1))*percent_retrace))
    cond3_bear = (close <= (close.shift(1) - (close.shift(1) - openn.shift(1))*percent_retrace))

    condbb_bull = (df['Low'] <= bb_low) | (df['Low'].shift(1) <= bb_low.shift(1))
    condbb_pos_bull = (df['High'] <= ma20)
    condsma50_bull = ((df['Low'] <= ma50) & (df['High'] >= ma50)) | ((df['Low'].shift(1) <= ma50.shift(1)) & (df['High'].shift(1) >= ma50.shift(1)))

    condbb_bear = (df['High'] >= bb_up) | (df['High'].shift(1) >= bb_up.shift(1))
    condbb_pos_bear = (df['Low'] >= ma20)
    condsma50_bear = ((df['High'] >= ma50) & (df['Low'] <= ma50)) | ((df['High'].shift(1) >= ma50.shift(1)) & (df['Low'].shift(1) <= ma50.shift(1)))

    df.loc[cond1 & cond2_bear & cond3_bear & condbb_pos_bear & (condbb_bear | condsma50_bear), 'tweezer_bear'] = True
    df.loc[cond1 & cond2_bull & cond3_bull & condbb_pos_bull & (condbb_bull | condsma50_bull), 'tweezer_bull'] = True
    cond4_bear = (df['tweezer_bear'].shift(1) == True) & (close <= df['Low'].shift(1))
    cond4_bull = (df['tweezer_bull'].shift(1) == True) & (close >= df['High'].shift(1))

    df.loc[cond4_bull, 'confirmed_bull'] = True
    df.loc[cond4_bear, 'confirmed_bear'] = True

    df_tw = df[(df['tweezer_bear'] == True) | (df['tweezer_bull'] == True)]
    df_tw_confirmed = df[(df['confirmed_bear'] == True) | (df['confirmed_bull'] == True)]
    return df['tweezer_bear'], df['tweezer_bull'], df['confirmed_bear'], df['confirmed_bull'], df_tw, df_tw_confirmed

def check_twz_bull(df, range_accept = 0.1/100, percent_retrace = 0.33/100) :
    twz_bear,twz_bull,confirmed_twz_bear,confirmed_twz_bull,df_tw,df_tw_confirmed = get_tweezer(df, range_accept = 0.1/100, percent_retrace = 0.33/100)
    state = 'off'
    if (twz_bull.iloc[-1] == True) :
        state = 'tweezer_bull'
    if (confirmed_twz_bull.iloc[-1] == True) :
        state = 'confirmed_tweezer_bull'
    return state

def check_twz_bear(df, range_accept = 0.1/100, percent_retrace = 0.33/100) :
    twz_bear, twz_bull, confirmed_twz_bear, confirmed_twz_bull,df_tw,df_tw_confirmed = get_tweezer(df, range_accept = 0.1/100, percent_retrace = 0.33/100)
    state = 'off'
    if (twz_bear.iloc[-1] == True) :
        state = 'tweezer_bear'
    if (confirmed_twz_bear.iloc[-1] == True) :
        state = 'confirmed_tweezer_bear'
    return state

def ao(df):
    # Calculate the median prices for each period
    df['median'] = (df['High'] + df['Low']) / 2

    # Calculate the SMA with 5 and 34 periods
    df['sma5'] = df['median'].rolling(window=5).mean()
    df['sma34'] = df['median'].rolling(window=34).mean()

    # Calculate the AO indicator by subtracting the SMA with 34 periods from the SMA with 5 periods
    df['ao'] = df['sma5'] - df['sma34']
    return df['ao']

def bob_ao(df):
    df['bob_ao'] = 'Neutral'
    df.loc[(df['ao']>df['ao'].shift(1)), 'bob_ao'] = 'Bullish'
    df.loc[(df['ao']<df['ao'].shift(1)), 'bob_ao'] = 'Bearish'
    return df['bob_ao']

def bull_bear_rsi_div(data) :
    if "rsi" not in data :
        data['rsi'] = RSI(data, 14)
    data = data.reset_index(drop=True)

    rsi_values = data['rsi'].values

    distance = 7
    tops = argrelextrema(rsi_values, np.greater, order=distance)
    bottoms = argrelextrema(rsi_values, np.less, order=distance)

    data['top_rsi'] = False
    for i in tops[0] :
        data['top_rsi'][i] = True
        #data.loc[data.index == i, 'top_rsi'] = True

    data['bot_rsi'] = False
    for i in bottoms[0] :
        data['bot_rsi'][i] = True
        #data.loc[data.index == i, 'top_rsi'] = True

    data_bottom = data[data['bot_rsi'] == True]
    data_bottom['bullish_rsi_div'] = None
    data_bottom.loc[(data_bottom['rsi'] >= data_bottom['rsi'].shift(1)) & (data_bottom['Close'] <= data_bottom['Close'].shift(1)), 'bullish_rsi_div'] = True

    data_top = data[data['top_rsi'] == True]
    data_top['bearish_rsi_div'] = None
    data_top.loc[(data_top['rsi'] <= data_top['rsi'].shift(1)) & (data_top['Close'] >= data_top['Close'].shift(1)), 'bearish_rsi_div'] = True
    return data, data_bottom, data_top

def compression(df) :
    window=20
    df["sma_high"]=df['High'].rolling(window).mean()
    df["sma_low"]=df['Low'].rolling(window).mean()

    #checking compression
    df["dif_sma"]=df["sma_high"]-df["sma_low"]
    df["dif_sma_rol3"]=df["dif_sma"].rolling(3).mean()
    df["delta_sma_rol3"] = df["dif_sma_rol3"]-df["dif_sma_rol3"].shift(1)
    delta_sma_rol3 = all(x < 0 for x in df["delta_sma_rol3"].values[-5:])

    #checking higher high
    df["sma_high_rol3"]=df['sma_high'].rolling(3).mean()
    df["delta_sma_high_rol3"]=df['sma_high_rol3']-df['sma_high_rol3'].shift(1)

    rising_sma_high_rol3 = all(x > 0 for x in df["delta_sma_high_rol3"].values[-5:])
    falling_sma_high_rol3 = all(x < 0 for x in df["delta_sma_high_rol3"].values[-5:])


    #checking higher low
    df["sma_low_rol3"]=df['sma_low'].rolling(3).mean()
    df["delta_sma_low_rol3"]=df['sma_low_rol3']-df['sma_low_rol3'].shift(1)

    rising_sma_low_rol3 = all(x > 0 for x in df["delta_sma_low_rol3"].values[-5:])
    falling_sma_low_rol3 = all(x < 0 for x in df["delta_sma_low_rol3"].values[-5:])

    if delta_sma_rol3 & rising_sma_high_rol3 & rising_sma_low_rol3 :
        return "rising compression 🐻"
    elif delta_sma_rol3 & falling_sma_high_rol3 & falling_sma_low_rol3 :
        return "falling compression 🐸"
    else :
        return None











"""
def checkdiv(df) :
    df1 = df.copy()
    # Drop unused columns
    df = df[["Date", "Close"]]
    price = data["Close"].values
    dates = data["Date"]
    # Get higher highs, lower lows, etc.
    order = 5
    hh = getHigherHighs(price, order)
    lh = getLowerHighs(price, order)
    ll = getLowerLows(price, order)
    hl = getHigherLows(price, order)
    # Get confirmation indices
    hh_idx = np.array([i[1] + order for i in hh])
    lh_idx = np.array([i[1] + order for i in lh])
    ll_idx = np.array([i[1] + order for i in ll])
    hl_idx = np.array([i[1] + order for i in hl])

    df['rsi'] = RSI(df, 20)
    rsi = data['RSI'].values
    rsi_hh = getHigherHighs(rsi, order)
    rsi_lh = getLowerHighs(rsi, order)
    rsi_ll = getLowerLows(rsi, order)
    rsi_hl = getHigherLows(rsi, order)

    rsi_hh_idx = getHHIndex(rsi, order)
    rsi_lh_idx = getLHIndex(rsi, order)
    rsi_ll_idx = getLLIndex(rsi, order)
    rsi_hl_idx = getHLIndex(rsi, order)


#pour les div
from scipy.signal import argrelextrema
from collections import deque

def getHigherHighs(data: np.array, order=5, K=2):
    # Get highs
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    # Ensure consecutive highs are higher than previous highs
    extrema = []
    ex_deque = deque(maxlen=K)
    for i, idx in enumerate(high_idx):
      if i == 0:
        ex_deque.append(idx)
        continue
      if highs[i] < highs[i-1]:
        ex_deque.clear()
      ex_deque.append(idx)
      if len(ex_deque) == K:
        extrema.append(ex_deque.copy())
    return extrema
"""
