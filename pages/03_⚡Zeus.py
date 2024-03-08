import streamlit as st
import yfinance as yf
import pandas as pd
import os
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from app.ta import ao, bob_ao, get_squeeze, get_kc

st.set_page_config(layout = 'wide')

from app.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])


import ta
import numpy as np
st.sidebar.caption("*NOT FINANCIAL ADVICE!! FOR EDUCATION ONLY*")
":zap:"
st.caption("_Zeus (/zjuːs/; Ancient Greek: Ζεύς)[a] is the sky and thunder god in ancient Greek religion and mythology, who rules as king of the gods on Mount Olympus. His name is cognate with the first syllable of his Roman equivalent Jupiter._")
""
""

col1, col2, col3 = st.columns(3)
market = col1.radio('Market', ['sp500', 'crypto'], index=1)

broker="binance"
if market == "crypto" :
    broker = col2.radio("broker", ["binance","coinbase"], index=1)

path = f'dataset/{market}_{broker}/' if market == "crypto" else f'dataset/{market}/'

tables = [x.replace(".parquet","") for x in os.listdir(path)]

# Create dropdown menu to select ticker
ticker = col1.selectbox("Select a ticker:", tables)
emoji = ":chart_with_upwards_trend:" if market == "sp500" else ":coin:"
f"# {emoji} {market} - {ticker}"

try:
    # Download data for selected ticker
    data = pd.read_parquet(f"{path}{ticker}.parquet")
except :
    st.error('Erreur')
    st.stop()

if broker == "coinbase" :
    data["order"] = data["Date"].str[-4:] + data["Date"].str[3:5] + data["Date"].str[:2]
    data["order"] = data["order"].astype(int)

if data.empty :
    st.error('empty table')
    st.stop()


window = 20
list_col = ["Close", "Open", "High", "Low", "Volume"]
for col in list_col :
    data[col] = data[col].astype(float)

if market == 'crypto' :
    data['Date'] = data['order'].astype(str).str[6:8] + '/' + data['order'].astype(str).str[4:6] + '/' + data['order'].astype(str).str[:4]



guess = st.button("Guess next day price")
if guess :
    "From 14 last price data (ohlc), try to predict close of day +1 (day0 ignored)"
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    data2 = data[["Close","Open","High","Low","Volume"]]
    list_col = ["Close","Open","High","Low","Volume"]
    for i in range(14) :
        for elem in list_col :
            data2[f"{elem}{i}"] = data2[f"{elem}"].shift(i)
    data2["Close+1"] = data2["Close"].shift(-1)

    X = data2.dropna().drop(columns=["Open","High","Low","Close","Close+1"])
    y = data2.dropna()["Close+1"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    pred = RandomForestRegressor()
    pred.fit(X_train, y_train)

    # st.write(f"Accuracy {mname}: {pred.score(X_test, y_test)}")
    y_pred = pred.predict(X_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers"))
    st.plotly_chart(fig)

    next_day_close = pred.predict(data2.drop(columns=["Open","High","Low","Close","Close+1"]).iloc[-4:])
    "Yesterday close: $", float(data["Close"].values[-2]), "Today's close: $", data["Close"].values[-1], "Predicted next day close: $", next_day_close[-1]









# Calculate moving average
data["ma"] = data["Close"].ewm(span=window, adjust=False).mean()
# Calculate RSI
data["rsi"] = ta.momentum.RSIIndicator(data["Close"], window=14).rsi()
# Calculate Dot
# Calculate "dot" and "trendline" indicators
data["dot"] = data["Close"].ewm(span=window, adjust=False).mean()
data["trendline"] = data["Close"].ewm(span=window, adjust=False).mean().ewm(span=window, adjust=False).mean()
data['ao'] = ao(data)
data['bob_ao'] = bob_ao(data)

# Determine trend based on "dot" and "trendline" indicators
data.loc[data["dot"] > data["trendline"], 'sentiment'] = 'bullish'
data.loc[data["dot"] < data["trendline"], 'sentiment'] = 'bearish'
data.loc[ (data["rsi"] > 40) & (data["rsi"] < 60) , "sentiment"] = np.nan

data.loc[data["sentiment"]=='bullish', 'dot_y'] = data['Low']
data.loc[data["sentiment"]=='bearish', 'dot_y'] = data['High']

data.loc[data['dot'] == 1, 'dot_y'] = data['Low']
data.loc[data['dot'] == -1, 'dot_y'] = data['High']

color_dict = {1:'green', -1:'red'}

data_len = len(data)
days = col3.slider("days to load", 2, data_len, 2000 if data_len>2000 else data_len)
data = data.tail(days)

#sqz
data["squeeze"] = get_squeeze(data,20)
data["kc_middle"], data["kc_upper"], data["kc_lower"] = get_kc(data, 20, 2, 20/2)

''
''
'### Board'
col1, col2, col3, col4 = st.columns(4)
if (data['Volume'].quantile(q=0.8) < data['Volume'].iloc[-1]) | (data['Volume'].quantile(q=0.8) < data['Volume'].iloc[-2]) :
    volume_status = '🔥 High'
elif (data['Volume'].quantile(q=0.25) <= data['Volume'].iloc[-1]) & (data['Volume'].quantile(q=0.75) >= data['Volume'].iloc[-1]) :
    volume_status = '🍃 Neutral'
else :
    volume_status = '💤 Low'
col1.write('##### Volume')
col1.write(volume_status)
#col1.write(str(data['Volume'].iloc[-1]))


rsi_boundaries=[30,70]
if data['rsi'].iloc[-1] < rsi_boundaries[0] :
    rsi_status = '🟢 Low (<30)'
elif (data['rsi'].iloc[-1] >= rsi_boundaries[0]) & (data['rsi'].iloc[-1] <= rsi_boundaries[1]) :
    rsi_status = '💤 Neutral'
else :
    rsi_status = '🔴 High (>70)'
col2.write('##### RSI')
col2.write(rsi_status)



ao_status = 'empty'
if data['bob_ao'].iloc[-1] == 'Bullish' :
    ao_status = '🟢 Bullish'
elif data['bob_ao'].iloc[-1] == 'Neutral' :
    ao_status = '💤 Neutral'
elif data['bob_ao'].iloc[-1] == 'Bearish' :
    ao_status = '🔴 Bearish'
col3.write('##### AO')
col3.write(ao_status)


col4.write('##### Squeeze')
col4.write(data["squeeze"].values[-1])


'#### Dot and trendline strategy'
# Create candlestick chart with "dot" and "trendline" indicators
fig = go.Figure(data=[go.Candlestick(x=data["Date"], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                                     increasing=dict(line=dict(color="palegreen")), decreasing=dict(line=dict(color="antiquewhite")))])
data_bear_dot = data[data['sentiment']=='bearish']
data_bull_dot = data[data['sentiment']=='bullish']

# for i in range(len(data_bear_dot)) :
#     row = data_bear_dot.iloc[i]
#     fig.add_trace(go.Scatter(x=[row['Date'], row['Date']], y=[row['High'], data['High'].max()], mode='lines', line_color='rgba(112, 0, 0, 0.3)', showlegend=False))
#
# for i in range(len(data_bull_dot)) :
#     row = data_bull_dot.iloc[i]
#     fig.add_trace(go.Scatter(x=[row['Date'], row['Date']], y=[row['Low'], data['Low'].min()], mode='lines', line_color='rgba(0, 112, 0, 0.3)', showlegend=False))

# colors_sqz=["rgba(0, 0, 0, 0.7)" if val != "off" else "rgba(0, 0, 255, 0.)" for val in data["squeeze"]]
colors_sqz=["sienna" if val != "off" else "rgba(0, 0, 255, 0.)" for val in data["squeeze"]]

fig.add_scatter(x=data[data['sentiment']=='bearish']["Date"], y=data[data['sentiment']=='bearish']["dot_y"], name="bear-dot", mode="markers", marker_color='red')
fig.add_scatter(x=data[data['sentiment']=='bullish']["Date"], y=data[data['sentiment']=='bullish']["dot_y"], name="bull-dot", mode="markers", marker_color='green', marker_size=8)
fig.add_scatter(x=data["Date"], y=data['trendline'], name="20d ema", mode='lines', line_width=1, line_color='yellow')
fig.add_scatter(x=data["Date"], y=data['dot'], name="20(20d ema) ema", mode='lines', line_width=1, line_color='dodgerblue')
fig.add_scatter(x=data["Date"], y=data["kc_upper"], mode="markers", marker={'color': colors_sqz, 'symbol':"triangle-down"}, name="kc upper squeeze")
fig.add_scatter(x=data["Date"], y=data["kc_lower"], mode="markers", marker={'color': colors_sqz, 'symbol':"triangle-up"}, name="kc lower squeeze")

fig.update_layout(height=650, template='simple_white', title_text=f"{ticker} SQUEEZE & DOT")#, plot_bgcolor="silver")
fig.update_xaxes(rangeslider_visible=False, title="Date")
fig.update_yaxes(title="Price")

# Show chart
st.plotly_chart(fig, use_container_width = True)
''
'#### Volume & RSI'
fig = px.scatter(data, x='Date', y='Close', color='Volume', color_continuous_scale='jet', size='Volume', opacity=0.5, title=ticker+' accumulation/distribution vs rsi', height=700)
data_bull_rsi = data[data['rsi']<rsi_boundaries[0]]
data_bear_rsi = data[data['rsi']>rsi_boundaries[1]]
# fig.add_trace(go.Scatter(x=data['Date'], y=data['rsi']*data['Close'].max()/100, mode='lines', line_color='yellow', name='rsi'))


# fig.add_trace(go.Scatter(x=data_bull_rsi['Date'], y=data_bull_rsi['rsi']*data['Close'].max()/100, mode='markers', marker_color='lawngreen', name='bull rsi'))
# fig.add_trace(go.Scatter(x=data_bear_rsi['Date'], y=data_bear_rsi['rsi']*data['Close'].max()/100, mode='markers', marker_color='crimson', name='bear rsi'))

def float_to_rgba_jet(column):
    normalized_column = (column - column.min()) / (column.max() - column.min())
    rgb_column = []
    for i in normalized_column :
        if i <= 0.125:
            R = 0
            G = 0
            B = 4 * i + 0.5
            A = 0.03
        elif i <= 0.375:
            R = 0
            G = 4 * (i - 0.125)
            B = 1
            A = 0.1
        elif i <= 0.625:
            R = 4 * (i - 0.375)
            G = 1
            B = 1 - 4 * (i - 0.375)
            A = 0.2
        elif i <= 0.875:
            R = 1
            G = 1 - 4 * (i - 0.625)
            B = 0
            A = 0.5
        else:
            R = 1 - 4 * (i - 0.875)
            G = 0
            B = 0
            A = 0.7
        R = int(R * 255)
        G = int(G * 255)
        B = int(B * 255)
        rgb_value = (R, G, B, A)
        rgb_column.append('rgba'+str(rgb_value))
    return rgb_column

data['volume_color'] = float_to_rgba_jet(data['Volume'])


for i in range(len(data)) :
    row = data.iloc[i]
    #fig.add_hline(y=row['Close'], line_width=1, line_color=row['volume_color'])
    fig.add_trace(go.Scatter(x=[row['Date'], data.iloc[-1]['Date']], y=[row['Close'], row['Close']], mode='lines', line_color=row['volume_color'], showlegend=False))
# for i in range(len(data_bull_rsi)) :
#     row = data_bull_rsi.iloc[i]
#     fig.add_trace(go.Scatter(x=[row['Date'], row['Date']], y=[row['Close'], data['Close'].min()-data['Close'].min()/10], mode='lines', line_color='rgba(0, 112, 0, 0.3)', showlegend=False if i > 0 else True, name=f"rsi<{rsi_boundaries[0]}"))
#
# for i in range(len(data_bear_rsi)) :
#     row = data_bear_rsi.iloc[i]
#     fig.add_trace(go.Scatter(x=[row['Date'], row['Date']], y=[row['Close'], data['Close'].max()+data['Close'].max()/10], mode='lines', line_color='rgba(112, 0, 0, 0.3)', showlegend=False if i > 0 else True, name=f"rsi>{rsi_boundaries[1]}"))


fig.update_layout(coloraxis_colorbar_x=-0.17, template='simple_white')
#fig.update_layout(coloraxis={'showscale': False}, template='simple_white')

st.plotly_chart(fig, use_container_width=True)

'#### Test rsi div'
fig = px.line(data, x='Date', y='Close', title=ticker+' rsi div', height=700)
window=5
data['bull_div'] = False
data.loc[(data['rsi'].rolling(window).mean() > data['rsi'].rolling(window).mean().shift(1)) & (data['Close'].rolling(window).mean() < data['Close'].rolling(window).mean().shift(1)), 'bull_div'] = True

data['bear_div'] = False
data.loc[(data['rsi'].rolling(window).mean() < data['rsi'].rolling(window).mean().shift(1)) & (data['Close'].rolling(window).mean() > data['Close'].rolling(window).mean().shift(1)), 'bear_div'] = True

data = data.dropna(subset=['rsi'])
data['div_color'] = float_to_rgba_jet(data['rsi'])


data_bear_div = data[data['bear_div']==True]
#data_bear_div['div_color'] = float_to_rgba_jet(data_bear_div['rsi'])
data_bull_div = data[data['bull_div']==True]
#data_bull_div['div_color'] = float_to_rgba_jet(data_bull_div['rsi'])


for i in range(len(data_bull_div)):
    row=data_bull_div.iloc[i]
    fig.add_trace(go.Scatter(x=[row['Date'], row['Date']], y=[row['Close'], data['Low'].min()], mode='lines', line_color=row['div_color'], showlegend=False))

for i in range(len(data_bear_div)):
    row=data_bear_div.iloc[i]
    fig.add_trace(go.Scatter(x=[row['Date'], row['Date']], y=[row['Close'], data['High'].max()], mode='lines', line_color=row['div_color'], showlegend=False))
fig.update_layout(template='simple_white')
st.plotly_chart(fig, use_container_width=True)


''
'#### AO'
data['Date_index'] = data.index
c_d_map = {'Bullish':'lightseagreen', 'Bearish':'crimson', 'Neutral':'grey'}
fig = px.scatter(data, x='Date_index', y='Close', color='bob_ao', color_discrete_map=c_d_map)
fig.add_trace(go.Scatter(x=data['Date_index'], y=data['Close'], mode='lines', line_color='grey'))


st.plotly_chart(fig, use_container_width=True)

def bull_bear_rsi_div(data) :
    import numpy as np
    from scipy.signal import argrelextrema
    data = data.reset_index(drop=True)

    rsi_values = data['rsi'].values

    distance = 9
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

data, data_bottom, data_top = bull_bear_rsi_div(data)

data['rsi'] = data['rsi'] / 100 * (data['Close'].min() + data['Close'].max())/2
data_bottom['rsi'] = data_bottom['rsi'] / 100 * (data['Close'].min() + data['Close'].max())/2
data_top['rsi'] = data_top['rsi'] / 100 * (data['Close'].min() + data['Close'].max())/2


fig = px.line(data, x='Date', y='Close').update_traces(line=dict(width=1))
fig.add_trace(go.Scatter(x=data['Date'], y=data['rsi'], mode='lines', line_width=1, name="rsi"))
for i in range(len(data_bottom)):
    row = data_bottom.iloc[i]
    prev_row = data_bottom.iloc[i-1]
    if row['bullish_rsi_div'] == True :
        x = [row['Date'], prev_row['Date']]
        y = [row['rsi'], prev_row['rsi']]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='limegreen', line_width=1, showlegend = False))

for i in range(len(data_top)):
    row = data_top.iloc[i]
    prev_row = data_top.iloc[i-1]
    if row['bearish_rsi_div'] == True :
        x = [row['Date'], prev_row['Date']]
        y = [row['rsi'], prev_row['rsi']]
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='crimson', line_width=1, showlegend = False))

data_bottom = data_bottom[data_bottom['bullish_rsi_div'] == True]
fig.add_trace(go.Scatter(x=data_bottom['Date'], y=data_bottom['rsi'], mode='markers', marker_color='limegreen', marker_size=10, name="bull div"))

data_top = data_top[data_top['bearish_rsi_div'] == True]
fig.add_trace(go.Scatter(x=data_top['Date'], y=data_top['rsi'], mode='markers', marker_color='crimson', marker_size=10, name="bear div"))

st.plotly_chart(fig, use_container_width=True)


# from plotly.subplots import make_subplots
# fig = make_subplots(rows=2, cols=1, row_heights=[0.8,0.2], shared_xaxes=True)
# fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], mode="lines", line_color="blue"), row=1, col=1)
# fig.add_trace(go.Scatter(x=data["Date"], y=data["rsi"], mode="lines", line_color="yellow"), row=2, col=1)
# for i in range(len(data_bottom)):
#     row = data_bottom.iloc[i]
#     prev_row = data_bottom.iloc[i-1]
#     if row['bullish_rsi_div'] == True :
#         x = [row['Date'], prev_row['Date']]
#         y = [row['rsi'], prev_row['rsi']]
#         fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='limegreen', line_width=1, showlegend = False), row=2, col=1)
#
# for i in range(len(data_top)):
#     row = data_top.iloc[i]
#     prev_row = data_top.iloc[i-1]
#     if row['bearish_rsi_div'] == True :
#         x = [row['Date'], prev_row['Date']]
#         y = [row['rsi'], prev_row['rsi']]
#         fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line_color='crimson', line_width=1, showlegend = False), row=2, col=1)
#
# st.plotly_chart(fig, use_container_width=True)

"#### GRM50 Dimensional addition"

data["gain"]=(data["Close"]-data["Open"])/data["Open"]
data=data.dropna(subset=["Close"])
data["ma50"]=100*(data["Close"]-data["Close"].rolling(50).mean().rolling(50).mean())/data["Close"]
data=data[["Date", "gain", "rsi", "ma50"]]
data["Date"]=data["Date"].astype(str)

list_prev_dates=[]
for i in range(5):
    list_prev_dates.append(str(data["Date"].values[-(i+1)]))



prev_data=[]
for date in list_prev_dates :
    prev_data.append(data[data["Date"]==date])

cl_list=["red", "orange", "yellow", "green", "cyan"]

fig=go.Figure()
fig.add_trace(go.Scatter3d(x=data["ma50"], y=data["rsi"], z=data["gain"], mode="lines", line_color="brown", name="trackline",
                           hovertemplate='<b>%{text}</b>',
                           text=data["Date"].astype(str).values))

for i in range(len(prev_data)):
    fig.add_trace(go.Scatter3d(x=prev_data[i]["ma50"], y=prev_data[i]["rsi"], z=prev_data[i]["gain"], marker_color=cl_list[i], name=list_prev_dates[i]))


fig.update_layout(height=800)

st.plotly_chart(fig,use_container_width=True)
