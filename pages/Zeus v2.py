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
st.caption("*NOT FINANCIAL ADVICE!! FOR EDUCATION ONLY*")
"# :zap: Zeus"
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


with st.expander("Plot options") :
    col1, col2, col3, col4, col5 = st.columns(5)
    MAs=col1.multiselect("Show moving averages", [6, 14, 20, 50, 200], None, placeholder="Choose MA to display")
    show_ema = col1.toggle("Show EMA")
    
fig=go.Figure()
fig.add_trace(go.Candlestick( x=data["Date"], open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                              increasing=dict(line=dict(color="palegreen")), decreasing=dict(line=dict(color="antiquewhite"))))

fig.update_layout(height=650, template='simple_white', title_text=f"{ticker} daily")
fig.update_xaxes(rangeslider_visible=False, title="Date")

st.plotly_chart(fig, use_container_width=True)
