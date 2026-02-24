import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import plotly.express as px
import plotly.graph_objects as go




def np_ewm(arr, span):
    alpha = 2 / (span + 1.0)
    ewm = np.empty_like(arr, dtype=float)
    ewm[0] = arr[0]
    for i in range(1, len(arr)):
        ewm[i] = alpha * arr[i] + (1 - alpha) * ewm[i-1]
    return ewm

def np_rsi(close, periods):
    close_delta = np.diff(close)
    up = np.where(close_delta > 0, close_delta, 0)
    down = np.where(close_delta < 0, -close_delta, 0)
    ma_up = np_ewm(up, periods - 1)
    ma_down = np_ewm(down, periods - 1)
    # rsi = np.where(ma_down != 0, 100 - (100 / (1 + ma_up / ma_down)), 100)
    condition = (ma_down != 0) & (ma_down > 0)
    rsi = np.full_like(ma_up, 100.0)  # Par défaut 100 si condition non remplie
    rsi[condition] = 100 - (100 / (1 + ma_up[condition] / ma_down[condition]))

    return rsi


def check_supertrend_flip(data, period=10, multiplier=3):
    """
    Calculate Supertrend indicator and detect bullish/bearish flips.
    Returns 'bullish' if the last candle flips to price above Supertrend,
    'bearish' if below, or None if no flip.

    Args:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns
        period (int): ATR period (default: 10)
        multiplier (float): Multiplier for ATR (default: 3)

    Returns:
        str: 'bullish', 'bearish', or None
    """
    # Calculate ATR
    data['tr'] = np.maximum(
        data['High'].values - data['Low'].values,
        np.maximum(
            np.abs(data['High'].values - data['Close'].shift(1).values),
            np.abs(data['Low'].values - data['Close'].shift(1).values)
        )
    )
    data['atr'] = pd.Series(data['tr'].rolling(window=period).mean().values)

    # Calculate basic upper and lower bands
    data['basic_ub'] = (data['High'].values + data['Low'].values) / 2 + multiplier * data['atr'].values
    data['basic_lb'] = (data['High'].values + data['Low'].values) / 2 - multiplier * data['atr'].values

    # Initialize Supertrend
    supertrend = np.full(len(data), np.nan)
    trend = np.zeros(len(data), dtype=int)

    # Set initial Supertrend value
    if len(data) > period:
        supertrend[period] = data['basic_ub'].values[period]

        # Calculate Supertrend vectorized where possible
        close_vals = data['Close'].values
        basic_ub_vals = data['basic_ub'].values
        basic_lb_vals = data['basic_lb'].values
        for i in range(period + 1, len(data)):
            if close_vals[i-1] <= supertrend[i-1]:
                supertrend[i] = min(basic_ub_vals[i], supertrend[i-1])
                trend[i] = 1
            else:
                supertrend[i] = max(basic_lb_vals[i], supertrend[i-1])
                trend[i] = -1

        data['supertrend'] = supertrend
        data['trend'] = trend

    # Detect flips
    if len(data) >= 2:
        current_trend = data['trend'].values[-1]
        prev_trend = data['trend'].values[-2]
        if current_trend == 1 and prev_trend == -1:
            return 'bullish'
        elif current_trend == -1 and prev_trend == 1:
            return 'bearish'

    return None

def check_macd_cross(data, fast=12, slow=26, signal=9):
    """
    Calculate MACD and detect bullish/bearish crosses.
    Returns 'bullish' if MACD crosses above signal line,
    'bearish' if below, or None if no cross.

    Args:
        data (pd.DataFrame): DataFrame with 'Close' column
        fast (int): Fast EMA period (default: 12)
        slow (int): Slow EMA period (default: 26)
        signal (int): Signal line period (default: 9)

    Returns:
        str: 'bullish', 'bearish', or None
    """
    # Calculate EMAs
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean().values
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean().values

    # Calculate MACD and Signal line
    macd = ema_fast - ema_slow
    signal_line = pd.Series(macd).ewm(span=signal, adjust=False).mean().values

    # Detect crosses
    if len(data) >= 2:
        if (macd[-2] <= signal_line[-2] and
            macd[-1] > signal_line[-1]):
            return 'bullish'
        elif (macd[-2] >= signal_line[-2] and
              macd[-1] < signal_line[-1]):
            return 'bearish'

    return None

def check_stochastic(data, k=14, d=3, smooth=3):
    """
    Calculate Stochastic Oscillator (%K) and return the latest %K value.

    Args:
        data (pd.DataFrame): DataFrame with 'High', 'Low', 'Close' columns
        k (int): Lookback period for %K (default: 14)
        d (int): Smoothing period for %D (default: 3)
        smooth (int): Smoothing period for %K (default: 3)

    Returns:
        float: Latest %K value
    """
    # Calculate lowest low and highest high
    lowest_low = data['Low'].rolling(window=k).min().values
    highest_high = data['High'].rolling(window=k).max().values

    # Calculate %K
    percent_k = 100 * (data['Close'].values - lowest_low) / (highest_high - lowest_low)

    # Smooth %K
    percent_k_smooth = pd.Series(percent_k).rolling(window=smooth).mean().values

    # Return latest %K value
    if not np.isnan(percent_k_smooth[-1]):
        return percent_k_smooth[-1]
    return np.nan


def df_check() :
    list_check = ['touching_MA','above_rsi', 'under_rsi', 'above_k', 'under_k',
                    'above_bb', 'under_bb', 'hu_mean', 'um', 'ham', 'squeeze',
                    'tweezer','divergence', 'compression', 'volume', "wick",
                    "dot","ao_breakout_up","ao_breakout_down"]
    df = pd.DataFrame({'ta_ref' : list_check, 'result' : [ '' for _ in list_check ]  })
    return df

def MA(df, window) :
    sma=df['Close'].rolling(window).mean()
    return sma

def touching_MA(df, window):
    df['MA'] = MA(df, window)
    ma_vals = df['MA'].values
    low_vals = df['Low'].values
    high_vals = df['High'].values
    state = False
    if (ma_vals[-1] >= low_vals[-1]) & (ma_vals[-1] <= high_vals[-1]) :
        state = True
    return state

def STD(df, window) :
    std = df['Close'].rolling(window).std()
    return std


def EMA(df, window) :
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return ema


def RSI(df, periods):
    close_delta = df['Close'].diff().values[1:]
    rsi_np = np_rsi(df['Close'].values, periods)
    rsi_series = pd.Series(np.full(len(df), np.nan), index=df.index)
    rsi_series.iloc[periods:] = rsi_np[-(len(df) - periods):]
    return rsi_series


def check_if_above_rsi(df, periods, ab_rsi_number) :
    if not "RSI" in df :
        df['RSI'] = RSI(df, periods)
    rsi_vals = df['RSI'].values
    state = False
    if rsi_vals[-1] >= ab_rsi_number :
        state = True
    return state

def check_if_under_rsi(df, periods, un_rsi_number) :
    if not "RSI" in df :
        df['RSI'] = RSI(df, periods)
    rsi_vals = df['RSI'].values
    state = False
    if rsi_vals[-1] <= un_rsi_number :
        state = True
    return state


def kd(df, n=34, x=5) :
    C  = df["Close"]
    Ln = df["Low"].rolling(n).min()
    Hn = df["High"].rolling(n).max()

    df["%K"] = 100 * ((C-Ln)/(Hn-Ln))

    Hx = (C-Ln).rolling(x).mean()
    Lx = (Hn-Ln).rolling(x).mean()

    df["%D"] = 100 * (Hx/Lx)

    return df["%K"], df["%D"]

def check_if_above_k(df, ab_k_number) :
    if not "%K" in df :
        df['%K'], _ = kd(df)
    k_vals = df['%K'].values
    state = False
    if k_vals[-1] >= ab_k_number :
        state = True
    return state

def check_if_under_k(df, un_k_number) :
    if not "%K" in df :
        df['%K'], _ = kd(df)
    k_vals = df['%K'].values
    state = False
    if k_vals[-1] <= un_k_number :
        state = True
    return state


def HU(df) :
    open_vals = df['Open'].values
    close_vals = df['Close'].values
    high_vals = df['High'].values
    low_vals = df['Low'].values

    body = high_vals - low_vals
    umb_ham = np.full(len(df), 'none', dtype=object)
    umb_cond = (open_vals >= (low_vals + body * 0.64)) & (close_vals >= (low_vals + body * 0.64))
    ham_cond = (open_vals <= (low_vals + body * 0.36)) & (close_vals <= (low_vals + body * 0.36))

    umb_ham[umb_cond] = 'umbrella'
    umb_ham[ham_cond] = 'hammer'

    return pd.Series(umb_ham, index=df.index)

def check_H(df) :
    hu = HU(df)
    hu_vals = hu.values
    state = False
    if len(hu) > 2 :
        if (hu_vals[-1] == 'hammer') or (hu_vals[-2] == 'hammer') :
            state = True
    return state

def check_U(df) :
    hu = HU(df)
    hu_vals = hu.values
    state = False
    if len(hu) > 2 :
        if (hu_vals[-1] == 'umbrella') or (hu_vals[-2] == 'umbrella') :
            state = True
    return state

def get_HU_mean(df, window) :
    df1 = df
    df1['umb_ham'] = HU(df1)
    umb_ham_vals = df1['umb_ham'].values
    open_vals = df1['Open'].values
    close_vals = df1['Close'].values
    high_vals = df1['High'].values
    low_vals = df1['Low'].values

    score_hu = np.zeros(len(df1))
    count_hu = np.zeros(len(df1), dtype=int)

    highest_oc = np.where(open_vals < close_vals, close_vals, open_vals)
    lowest_oc = np.where(open_vals > close_vals, close_vals, open_vals)

    count_hu[umb_ham_vals == 'umbrella'] = 1
    count_hu[umb_ham_vals == 'hammer'] = -1
    count_hu_series = pd.Series(count_hu)
    count_hu_roll = count_hu_series.rolling(window).sum()

    score_hu[umb_ham_vals == 'umbrella'] = 1 * (lowest_oc[umb_ham_vals == 'umbrella'] - low_vals[umb_ham_vals == 'umbrella'])
    score_hu[umb_ham_vals == 'hammer'] = -1 * (high_vals[umb_ham_vals == 'hammer'] - highest_oc[umb_ham_vals == 'hammer'])
    score_hu_series = pd.Series(score_hu)
    mean_hu = score_hu_series.rolling(window).sum()

    return mean_hu, count_hu_roll

def check_HU_mean(df, window) :
    mean_hu, count_hu = get_HU_mean(df, window)
    mean_hu_vals = mean_hu.values
    count_hu_vals = count_hu.values
    close_vals = df['Close'].values
    val = 1/10
    state = str(0)
    if (mean_hu_vals[-1] >= (close_vals[-1])*val) & (count_hu_vals[-1] >= 5) :
        emoji = ' 🐸'
        if (count_hu_vals[-1] >= 8) :
            emoji = ' 🐮'
        state = str(mean_hu_vals[-1]) + '_' + str(count_hu_vals[-1]) + emoji
    elif (mean_hu_vals[-1] <= (-close_vals[-1])*val) & (count_hu_vals[-1] <= -5) :
        emoji = ' 🐻'
        if (count_hu_vals[-1] <= -8) :
            emoji = ' 🐻‍❄️'
        state = str(mean_hu_vals[-1]) + '_' + str(count_hu_vals[-1]) + emoji
    return state


def get_bollinger_bands(df, window):
    sma = MA(df, window)
    std = STD(df, window)
    bb_up = sma + std * 2  # Calculate top band
    bb_low = sma - std * 2  # Calculate bottom band
    return bb_up, bb_low

def check_above_bband(df, window) :
    df['bb_up'], df['bb_down'] = get_bollinger_bands(df, window)
    bb_up_vals = df['bb_up'].values
    high_vals = df['High'].values
    state = False
    if high_vals[-1] >= bb_up_vals[-1] :
        state = True
    return state

def check_under_bband(df, window) :
    df['bb_up'], df['bb_down'] = get_bollinger_bands(df, window)
    bb_down_vals = df['bb_down'].values
    low_vals = df['Low'].values
    state = False
    if low_vals[-1] <= bb_down_vals[-1] :
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
    col_sqz_vals = col_sqz.values
    state = 'off'
    if (col_sqz_vals[-1] == 'on 🧨') | (col_sqz_vals[-1] == 'released 💥') :
        state = col_sqz_vals[-1]
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
    twz_bull_vals = twz_bull.values
    confirmed_twz_bull_vals = confirmed_twz_bull.values
    state = 'off'
    if (twz_bull_vals[-1] == True) :
        state = 'tweezer_bull'
    if (confirmed_twz_bull_vals[-1] == True) :
        state = 'confirmed_tweezer_bull'
    return state

def check_twz_bear(df, range_accept = 0.1/100, percent_retrace = 0.33/100) :
    twz_bear, twz_bull, confirmed_twz_bear, confirmed_twz_bull,df_tw,df_tw_confirmed = get_tweezer(df, range_accept = 0.1/100, percent_retrace = 0.33/100)
    twz_bear_vals = twz_bear.values
    confirmed_twz_bear_vals = confirmed_twz_bear.values
    state = 'off'
    if (twz_bear_vals[-1] == True) :
        state = 'tweezer_bear'
    if (confirmed_twz_bear_vals[-1] == True) :
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

    data.loc[:, 'top_rsi'] = False
    for i in tops[0] :
        # data['top_rsi'][i] = True
        data.loc[i, 'top_rsi'] = True
        #data.loc[data.index == i, 'top_rsi'] = True

    data.loc[:, 'bot_rsi'] = False
    for i in bottoms[0] :
        # data['bot_rsi'][i] = True
        data.loc[i, 'bot_rsi'] = True
        #data.loc[data.index == i, 'top_rsi'] = True

    data_bottom = data[data['bot_rsi'] == True]
    data_bottom['bullish_rsi_div'] = [ None for y in range(len(data_bottom))]
    data_bottom.loc[(data_bottom['rsi'] >= data_bottom['rsi'].shift(1)) & (data_bottom['Close'] <= data_bottom['Close'].shift(1)), 'bullish_rsi_div'] = True


    data_top = data[data['top_rsi'] == True]
    data_top['bearish_rsi_div'] = [None for y in range(len(data_top))]
    data_top.loc[(data_top['rsi'] <= data_top['rsi'].shift(1)) & (data_top['Close'] >= data_top['Close'].shift(1)), 'bearish_rsi_div'] = True
    return data, data_bottom, data_top

def compression(df) :
    window=20
    high_vals = df['High'].values
    low_vals = df['Low'].values
    sma_high = pd.Series(high_vals).rolling(window).mean().values
    sma_low = pd.Series(low_vals).rolling(window).mean().values

    #checking compression
    dif_sma = sma_high - sma_low
    dif_sma_rol3 = pd.Series(dif_sma).rolling(3).mean().values
    delta_sma_rol3 = dif_sma_rol3 - np.roll(dif_sma_rol3, 1)
    delta_sma_rol3[0] = 0  # First value NaN replacement
    delta_sma_rol3 = all(x < 0 for x in delta_sma_rol3[-5:])

    #checking higher high
    sma_high_rol3 = pd.Series(sma_high).rolling(3).mean().values
    delta_sma_high_rol3 = sma_high_rol3 - np.roll(sma_high_rol3, 1)
    delta_sma_high_rol3[0] = 0
    rising_sma_high_rol3 = all(x > 0 for x in delta_sma_high_rol3[-5:])
    falling_sma_high_rol3 = all(x < 0 for x in delta_sma_high_rol3[-5:])

    #checking higher low
    sma_low_rol3 = pd.Series(sma_low).rolling(3).mean().values
    delta_sma_low_rol3 = sma_low_rol3 - np.roll(sma_low_rol3, 1)
    delta_sma_low_rol3[0] = 0
    rising_sma_low_rol3 = all(x > 0 for x in delta_sma_low_rol3[-5:])
    falling_sma_low_rol3 = all(x < 0 for x in delta_sma_low_rol3[-5:])

    if delta_sma_rol3 & rising_sma_high_rol3 & rising_sma_low_rol3 :
        return "rising compression 🐻"
    elif delta_sma_rol3 & falling_sma_high_rol3 & falling_sma_low_rol3 :
        return "falling compression 🐸"
    else :
        return None





def predict_10(df1) :
    list_col = ["Open","High","Low","Close","Volume","RSI"]
    df1["RSI"] = RSI(df1,14)
    df = df1[list_col]
    for i in range(1,11) :
        for elem in ["Open","High","Low","Close","Volume","RSI"] :
            df[f"{elem}_{i}"] = df[elem].shift(i)
            list_col.append(f"{elem}_{i}")

    df = df.dropna(subset=list_col).reset_index(drop=True)
    y_col = []
    for i in range(1,8) :
        df[f"Close+{i}"] = df["Close"].shift(-i)
        y_col.append(f"Close+{i}")

    df2 = df = df.dropna(subset=y_col)


    X_train, X_test, y_train, y_test = train_test_split(df[list_col].values, df[y_col].values, test_size=0.2)
    val, r2_res, mse_res = [], [], []

    # for i in range(2,20) :
    all_res =[]
    for i in range(20) :
        regr = RandomForestRegressor(n_estimators=6, oob_score=True, max_depth=5, max_leaf_nodes=5)
        regr.fit(X_train, y_train)
        pred_test = regr.predict(X_test)
        mse = mean_squared_error(y_test, pred_test)
        # st.write(f'Mean Squared Error: {mse}')
        r2 = r2_score(y_test, pred_test)
        # st.write(f'R-squared: {r2}')
        # val.append(i)
        # r2_res.append(r2)
        # mse_res.append(mse)
        # fig = px.line(x=val, y=r2_res)
        # st.plotly_chart(fig)

        all_res.append(regr.predict(df2[list_col].tail(2)))
    # st.dataframe(all_res)
    return all_res

def pearson_rsi(data, grid_x, grid_type, gs_multiwindow):

    data["RSI"] = RSI(data, 14).values

    if grid_type == "Volume cum":
        if gs_multiwindow :
            col_names = []
            for i in range(2,202,10) :
                cn = f"{grid_type}_{i}"
                if len(data)>i :
                    data[cn] = data["Volume"].rolling(i).sum()
                    col_names.append(cn)
                else :
                    break
            data[grid_type] = data[col_names].mean(axis=1)
            col_names_count = []
            for cn in col_names :
                data.loc[data[cn]>data[cn].quantile(0.9), cn+"_count"] = 1
                data[cn+"_count"] = data[cn+"_count"].fillna(0)
                col_names_count.append(cn)
            data[grid_type + "_count"] = data[col_names_count].sum(axis=1)
            data[grid_type + "_count"] = data[grid_type + "_count"].fillna(0)

            data[grid_type + "_perc"] = data[grid_type + "_count"]/data[grid_type + "_count"].max()

        else :
            data[grid_type] = data["Volume"].rolling(grid_x).sum()
    else :
        # Calcul de la corrélation glissante (fenêtre de 20 périodes)
        if gs_multiwindow :
            col_names = []
            for i in range(4,100,5) :
                cn = f"{grid_type}_{i}"
                if len(data)>i :
                    data[cn] = data['Close'].rolling(i, min_periods=1).corr(data["RSI"])
                    col_names.append(cn)
                else :
                    break
            data[grid_type] = data[col_names].mean(axis=1)
            col_names_count = []
            for cn in col_names :
                data.loc[data[cn]<=0, cn+"_count"] = 1
                data[cn+"_count"] = data[cn+"_count"].fillna(0)
                col_names_count.append(cn)
            data[grid_type + "_count"] = data[col_names_count].sum(axis=1)
            data[grid_type + "_count"] = data[grid_type + "_count"].fillna(0)
            data[grid_type + "_perc"] = data[grid_type + "_count"]/data[grid_type + "_count"].max()
        else :
            window = grid_x
            data[grid_type] = data['Close'].rolling(window, min_periods=1).corr(data["RSI"])

    return data



def detect_star(df):
    # C1 (il y a 2 bougies)
    c1_open = df['Open'].shift(2) ; c1_close = df['Close'].shift(2)
    c1_body = (c1_open - c1_close).abs() # Positif si baissier
    c1_wick = df['High'].shift(2) - df['Low'].shift(2)


    # C2 (il y a 1 bougie)
    c2_open = df['Open'].shift(1) ; c2_close = df['Close'].shift(1)
    c2_body = (c2_open - c2_close).abs()


    # C3 (bougie actuelle)
    c3_open = df['Open'] ; c3_close = df['Close']
    c3_body = (c3_close - c3_open).abs() # Positif si haussier
    c3_wick = df['High'] - df['Low']

    # 1. Couleurs : C1 rouge, C3 verte
    cond_colors_morning = (c1_close < c1_open) & (c3_close > c3_open)
    cond_colors_evening = (c1_close > c1_open) & (c3_close < c3_open)


    # 2. Taille : C2 doit être une petite bougie (ex: < 30% de C1)
    cond_small_star = c2_body < (c1_body * 0.3)

    # 3. Position : C2 doit se situer sous la clôture de C1 (gap ou point bas)
    cond_star_pos_morning = (c2_open <= c1_close) & (c2_close <= c1_close) & ((c2_close <= c3_open) | (c2_open <= c3_open))
    cond_star_pos_evening = (c2_open >= c1_close) & (c2_close >= c1_close) & ((c2_close >= c3_open) | (c2_open >= c3_open))

    cond_range = (c1_body >= (c1_wick * 0.4)) & (c3_body >= (c3_wick * 0.6))


    # 4. Force du retournement : C3 doit clôturer au-dessus du milieu de C1
    # Formule du milieu de C1 : (Open + Close) / 2
    c1_midpoint = (c1_open-c1_close)*0.5+c1_close #(c1_open + c1_close) / 2
    cond_strong_reversal_morning = c3_close > c1_midpoint
    cond_strong_reversal_evening = c3_close < c1_midpoint

    # 5. confirm
    cond_confirm_morning = (df["Close"].shift(-1) > df['High'].shift(2)) | (df["Close"].shift(-2) > df['High'].shift(2))
    cond_confirm_evening = (df["Close"].shift(-1) < df['Low'].shift(2))  | (df["Close"].shift(-2) < df['Low'].shift(2))

    cond_ms           = cond_colors_morning & cond_small_star & cond_star_pos_morning & cond_strong_reversal_morning & cond_range
    cond_ms_confirmed = cond_ms & cond_confirm_morning

    cond_es           = cond_colors_evening & cond_small_star & cond_star_pos_evening & cond_strong_reversal_evening & cond_range
    cond_es_confirmed = cond_es & cond_confirm_evening


    # --- Combinaison finale ---
    df["morning_star"] = None ; df["evening_star"] = None

    df.loc[cond_ms, "morning_star"] = "MS" ; df.loc[cond_ms_confirmed, "morning_star"] = "MSC"
    df.loc[cond_es, "evening_star"] = "ES" ; df.loc[cond_es_confirmed, "evening_star"] = "ESC"

    #propagating to the 2 other candles, c1 and c2
    df.loc[df["morning_star"].shift(-1).notna(), "morning_star"] = df["morning_star"].shift(-1)
    df.loc[df["morning_star"].shift(-1).notna(), "morning_star"] = df["morning_star"].shift(-1)

    #propagating to the 2 other candles, c1 and c2
    df.loc[df["evening_star"].shift(-1).notna(), "evening_star"] = df["evening_star"].shift(-1)
    df.loc[df["evening_star"].shift(-1).notna(), "evening_star"] = df["evening_star"].shift(-1)


    return df
