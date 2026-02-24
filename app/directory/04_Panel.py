import streamlit as st
import yfinance as yf
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from functions.ta import ao, bob_ao, get_squeeze, get_kc, HU, RSI, pearson_rsi, detect_star
import functions.pathfunc as pf
from datetime import datetime

from functions.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])
st.set_page_config(layout = 'wide')


import numpy as np
from scipy.stats import pearsonr








st.sidebar.caption("*NOT FINANCIAL ADVICE!! FOR EDUCATION ONLY*")
"# :material/bolt: PANEL"
st.caption("_Zeus (/zjuːs/; Ancient Greek: Ζεύς)[a] is the sky and thunder god in ancient Greek religion and mythology, who rules as king of the gods on Mount Olympus. His name is cognate with the first syllable of his Roman equivalent Jupiter._")
""
c1,c2=st.sidebar.columns(2)
market = c1.radio('Market', ['stocks', 'crypto'], index=0, disabled=True)



if market == "stocks" :
    stocks_path = os.path.join(pf.get_path_data(), "stocks")
    submarkets = os.listdir(stocks_path)

    stock_list = pd.read_csv("data/etoro_stocks_with_keywords.csv", sep=",")

    all_kw = [ x.split("+") for x in stock_list["keywords"].values ]

    res = []
    for kw in all_kw :
        res.extend(kw)

    res = [ x.strip() for x in res ]
    all_kw = np.unique(res)

    kw = st.selectbox("Select a category", all_kw)

    stock_list = stock_list[(stock_list["keywords"].values == kw) | (stock_list["keywords"].str.contains(kw+" ")) | (stock_list["keywords"].str.contains(" "+kw))].reset_index(drop=True)
    stock_list

    lg = st.toggle("lighter graph")
    last_n_values = st.radio("Show last n Close", [50, 100, 200, 365, 1000], index=1, horizontal=True)

    nb_col = 6
    cs = st.columns(nb_col)
    v=0
    for tick,sm,ind in zip(stock_list["symbol"].values, stock_list["exchange"].values, stock_list["industry"].values) :
        if sm is None or sm == "-" :
            filepath=os.path.join("data","stocks",ind,str(tick)+".parquet")
        else :
            filepath=os.path.join("data","stocks",sm,str(tick)+".parquet")

        if os.path.exists(filepath) :
            df = pd.read_parquet(filepath)
            df = df.tail(last_n_values if len(df) >= last_n_values else len(df))
            if lg :
                df = df[df.index % 3 == 0]
            close = df["Close"].values
            cs[v].metric(tick, round(close[-1],3), delta=round((close[-1]-close[0])/close[0]*100,2), chart_data=close, chart_type="line", border=True )

            v+=1

            if v == nb_col :
                v=0
