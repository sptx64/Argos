import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import datetime
import os
import requests
import json
import ast
import time

import functions.pathfunc as pf

st.set_page_config(layout = 'centered')

# import urllib.request


# Placeholder for logging (adjust as per your actual implementation)
from functions.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])

"# :material/cloud: Hermes"
st.caption("""  _Hermes (/ˈhɜːrmiːz/; Greek: Ἑρμῆς) is an Olympian deity in ancient Greek religion and mythology considered the herald of the gods.
He is also considered the protector of human heralds, travelers, thieves, merchants, and orators. He is able to move quickly and freely between the worlds of the mortal and the divine aided by his winged sandals. Hermes plays the role of the psychopomp or "soul guide"—a conductor of souls into the afterlife.
In myth, Hermes functions as the emissary and messenger of the gods, and is often presented as the son of Zeus and Maia, the Pleiad. He is regarded as "the divine trickster", about which the Homeric Hymn to Hermes offers the most well-known account._
""")
st.caption("Better to update crypto database at 12:00 UTC (=daily candle close). Binance=Binance US compatible coins. Update speed: Binance=fast, SP500+Euronext=normal, Yahoo Finance=normal.")

# Binance data retrieval function
def import_crypto(ticker, tframe, start_date=None, api_id=3,):
    apis = [
        "https://api.binance.com", "https://api-gcp.binance.com", "https://api1.binance.com",
        "https://api.binance.us", "https://api2.binance.com", "https://api3.binance.com",
        "https://api4.binance.com"
    ]
    url = f"{apis[api_id]}/api/v3/klines?symbol={ticker}&interval={tframe}"
    payload = {}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    data = ast.literal_eval(response.text)

    df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Timestamp_end", "Amount", "Count", "Open_interest", "Turnover", "Taker_buy_base_asset_volume"])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['order'] = df['Date'].dt.strftime('%Y%m%d')
    df['order'] = df['order'].astype(int)
    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    df['Date'] = df['Date'].astype(str).str[:10]
    df['timeframe'] = tframe
    df = df.sort_values(by=['order'])
    return df


# Main UI
col1, col2, col3 = st.columns(3)
market = col1.radio('Market', ['stocks', 'crypto'], horizontal=True)

if market == 'stocks':
    boc = "binance"

    list_submarkets = sorted([ x.replace(".csv","") for x in os.listdir(os.path.join(pf.get_path_data(), "tickers list")) ])
    submarkets = st.pills("Sub market", list_submarkets, selection_mode = "multi")
    disable_upload = False
    if len(submarkets) == 0 :
        disable_upload = True


    tick_dfs = []
    for sm in submarkets :
        tick_dfs.append(pd.read_csv(os.path.join(pf.get_path_data(), "tickers list", sm + ".csv")))

        dataset_path = os.path.join(pf.get_path_data(), "stocks", sm)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

    if len(tick_dfs)>0 :
        tick_df = pd.concat(tick_dfs)
        st.caption(f":blue-badge[{len(tick_df)}] stocks are going to be updated.")


elif market == 'crypto':
    boc = col2.radio("Crypto broker to update", ["binance",])
    if boc == "binance":
        api = col3.selectbox("API", [0, 1, 2, 3, 4, 5, 6], index=3, help="Usually api 3 is available from anywhere without IP lock, but with limited tickers available. Other API may lock depending on the IP server but with a thousand of tickers.")
        apis = [
            "https://api.binance.com", "https://api-gcp.binance.com", "https://api1.binance.com",
            "https://api.binance.us", "https://api2.binance.com", "https://api3.binance.com",
            "https://api4.binance.com"
        ]
        st.write("Selected API for Binance:", apis[api])

    disable_upload = False
    dataset_path = pf.get_path_crypto()
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

today_utc = datetime.datetime.utcnow().strftime('%Y-%m-%d')
st.caption(f"Today's UTC date: **:blue-badge[{today_utc}]**")

with st.expander("Last update") :
    update_df_path = os.path.join(pf.get_path_data(), "update_tracker.parquet")
    update_df = pd.read_parquet(update_df_path)
    update_df

c1, c2, c3, c4, c5 = st.columns(5)
update = c1.button('Update', type="primary")

# Update logic
if update:
    if (market == "crypto") and (boc == "binance"):
        if os.path.exists(dataset_path):
            list_files = os.listdir(dataset_path)
            for file in list_files:
                os.remove(os.path.join(dataset_path, file))

    my_toast = st.toast( f"Updating...", icon=":material/hourglass_top:", duration="infinite")
    if market == 'stocks':
        stocks_bar = st.progress(0., "Updating stocks")
        len_all_tick = len(tick_df) ; value = 1

        for sm in submarkets :
            dataset_path = os.path.join(pf.get_path_data(), "stocks", sm)
            tick_df_sm = tick_df[tick_df["exchange"] == sm]

            #premature skipping if tickers lenght == 0
            if len(tick_df_sm) == 0 :
                continue

            sm_bar = st.progress(0., f"Updating {sm} stocks")

            tickers = tick_df_sm["symbol"].values
            len_sm_tick = len(tickers); value_sm = 1


            for tick in tickers:
                path_to_file = os.path.join(dataset_path, tick + ".parquet")
                data = yf.Ticker(tick).history(period="20y", interval='1d').reset_index()
                if not data.empty :
                    if "Dividends" in data :
                        data["Dividends"] = data["Dividends"].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                        data["Dividends"] = pd.to_numeric(data["Dividends"].values, errors="coerce")
                    data.to_parquet(path_to_file, compression="brotli")

                stocks_bar.progress(value/len_all_tick, f"{value}/{len_all_tick} - updating {sm} stock market...")
                sm_bar.progress(value_sm/len_sm_tick, f"{value_sm}/{len_sm_tick} - {tick}")
                my_toast.toast( f"Updated {tick}", icon=":material/check_small:", duration="infinite")

                value += 1; value_sm+=1
                if value % 20 == 0 :
                    time.sleep(0.5)
                if value % 200 == 0 :
                    time.sleep(3)
                if value % 1000 == 0 :
                    time.sleep(10)

            #cleaning source tickers
            loaded_tickers = [ x.replace(".parquet","") for x in os.listdir(dataset_path) ]
            initial_tickers = pd.read_csv(os.path.join(pf.get_path_data(), "tickers list", sm + ".csv"))
            initial_tickers = initial_tickers[initial_tickers["symbol"].isin(loaded_tickers)].reset_index(drop=True)
            initial_tickers.to_csv(os.path.join(pf.get_path_data(), "tickers list", sm + ".csv"), index=False)

            st.toast(f"{sm} ticker csv source cleaned with available stocks")

            update_df.loc[update_df["exchange"] == sm, "last_complete_update"] = today_utc
            update_df.to_parquet(update_df_path)
            sm_bar.empty()
        stocks_bar.empty()








    elif market == 'crypto':
        if boc == "binance":
            my_bar = st.progress(0., "Binance updating datas")
            tickers_data = requests.request("GET", f"{apis[api]}/api/v3/exchangeInfo")
            tickers_data = json.loads(tickers_data.content)
            tickers_data = tickers_data["symbols"]
            tickers = [x["symbol"] for x in tickers_data]
            status = [x["status"] for x in tickers_data]
            len_tick = len([x for x, y in zip(tickers, status) if y == "TRADING"])
            value = 1
            for tick, stat in zip(tickers, status):
                path_to_file = os.path.join(dataset_path, tick + ".parquet")
                if stat != "TRADING":
                    if os.path.exists(path_to_file):
                        os.remove(path_to_file)
                        continue
                if stat == "TRADING":
                    data = import_crypto(tick, '1d')
                    if len(data) > 0:
                        data.to_parquet(path_to_file, compression="brotli")
                    my_bar.progress(value / len_tick, f"Binance {value}/{len_tick} {tick}")
                    my_toast.toast( f"Updated {tick}", icon=":material/check_small:", duration="infinite")

                    value += 1

            my_bar.empty()

    my_toast.toast("Update done", icon=":material/check_small:", duration="short")
    st.badge("Database update done", color="green", icon=":material/check_small:")
