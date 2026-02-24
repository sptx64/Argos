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



def to_weekly(data) :
    data['week'] = pd.to_datetime(data['Date'], format='%d/%m/%Y').dt.isocalendar().week.astype(str)
    data["week"] = [ w if len(w)>1 else "0"+w for w in data["week"].values ]
    data['year'] = pd.to_datetime(data['Date'], format='%d/%m/%Y').dt.isocalendar().year.astype(str)
    data["week"] = [ y + "_" + w for w,y in zip(data["week"].values, data["year"].values)]

    aggregation = {
        'Date': 'first',
        'Low': 'min',
        'High': 'max',
        'Open': 'first',
        'Close': 'last',
        'Volume': 'sum'
    }

    if "order" in data :
        aggregation["order"] = "first"

    data = data.groupby("week").agg(aggregation).reset_index()
    data = data.sort_values(by="week").reset_index(drop=True)
    return data





"# :material/bolt: CORRELATIONS"

col1, col2, col3 = st.columns(3)
market = col1.radio('Market', ['stocks', 'crypto'], horizontal=True, disabled=True)
if market == "stocks" :
    list_submarkets = sorted([ x.replace(".csv","") for x in os.listdir(os.path.join(pf.get_path_data(), "tickers list")) ])
    submarkets = st.pills("Sub market", list_submarkets, selection_mode = "multi")
    disable_search = False
    if len(submarkets) == 0 :
        disable_search = True

    tickers = [] ; tickers_path = []
    for sm in submarkets :
        tickers.extend(os.listdir(os.path.join(pf.get_path_data(), "stocks", sm )))
        tickers_path.extend( [ os.path.join(pf.get_path_data(), "stocks", sm, x) for x in os.listdir(os.path.join(pf.get_path_data(), "stocks", sm )) ] )

    if len(tickers)>0 :
        st.caption(f":blue-badge[{len(tickers)}] stocks are going to be screened.")



    if len(submarkets) == 0 :
        st.stop()
    c1,c2 = st.columns(2)
    slct_tick = c1.selectbox("Stock correlation", tickers, format_func=lambda x : x.replace(".parquet",""))
    method = c2.selectbox("Method", ["Volume","Close"])

    data = pd.read_parquet(tickers_path[tickers.index(slct_tick)])
    data = to_weekly(data)

    # data["return"] = data["Close"].values / data["Close"].shift(1).values
    window = 6

    if method == "Volume" :
        data["SMA20_asset"] = data["Volume"].rolling(window).mean().astype(np.float32)
    else :
        data["SMA20_asset"] = data["Close"].rolling(window).mean().astype(np.float32)
    data = data[["week","SMA20_asset"]]

    if st.button("Go") :
        pb = st.progress(0., "Scanning ...")
        v=0; len_tickers = len(tickers)

        res = {"tick" : [], "corr" : []}
        for tick, tick_p in zip(tickers, tickers_path) :
            pb.progress(v/len_tickers, f"Scanning ... {tick.replace('.parquet','')}")
            v+=1
            if tick == slct_tick :
                continue


            df = pd.read_parquet(tick_p)
            df = to_weekly(df)

            if len(df) < 50 :
                continue

            # df["return"] = df["Close"] / df["Close"].shift(1)
            # df["True price"] = (df["Close"] + df["High"] + df["Low"])/3
            if method == "Volume" :
                df["SMA20"] = df["Volume"].rolling(window).mean().astype(np.float32)
            else :
                df["SMA20"] = df["Close"].rolling(window).mean().astype(np.float32)

            df = df[["week","SMA20"]]

            df_conc = pd.merge( data, df, on="week", how="left" ).dropna(subset=["SMA20_asset","SMA20"])
            # if tick == slct_tick :
            #     df_conc

            spearman_corr = df_conc['SMA20_asset'].corr(df_conc['SMA20'], method='spearman')
            res["tick"].append(tick)
            res["corr"].append(spearman_corr)
        pb.empty()


        res = pd.DataFrame(res).sort_values(by="corr").reset_index(drop=True)

        fig = px.bar(res, x="tick", y="corr", color="corr", color_continuous_scale="RdBu", range_color=[-1, 1], title=f"Correlation to {tick.replace('.parquet','')}",)
        st.plotly_chart(fig)

        with st.expander("Full computed file") :
            res
