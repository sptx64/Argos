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
market = col1.radio('Market', ['sp500', 'crypto'], horizontal=True)

rate=col1.radio("rate **only for yahoo finance**", ["permissive","restrictive"], help="to accomodate with yahoo finance limit rate, restrictive adds waiting time")

time_wait = [0.25, 1, 5, 10]

if rate=='restrictive' :
    multiple = st.number_input("multiply wait time by :", 1.0, 100.0, 2.0)
    time_wait = [ x * multiple for x in time_wait]

if market == 'sp500':
    boc = "binance"
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

    dataset_path = pf.get_path_crypto()

dataset_path = pf.get_path_sp500() if market == "sp500" else pf.get_path_crypto()
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

today_utc = datetime.datetime.utcnow().strftime('%Y-%m-%d')
c1, c2, c3, c4, c5 = st.columns(5)
update = c1.button('Update', type="primary")
st.caption(f"Today's UTC date: {today_utc}")

# Update logic
if update:
    if (market == "crypto") and (boc == "binance"):
        if os.path.exists(dataset_path):
            list_files = os.listdir(dataset_path)
            for file in list_files:
                os.remove(os.path.join(dataset_path, file))

    my_toast = st.toast( f"Updating...", icon=":material/hourglass_top:", duration="infinite")
    if market == 'sp500':
        my_bar = st.progress(0., "SP500 updating datas")
        # my_toast = st.toast( f"Updating...", icon=":material/hourglass_top:" )

        # tickers = pd.read_html(req)[0
        tickers = pd.read_csv(os.path.join(pf.get_path_app(), "sp500_companies.csv"))["Symbol"].values

        #below have been blocked by wikipedia
        # tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        # tickers = tickers['Symbol'].values
        if os.path.exists(os.path.join(pf.get_path_app(), "Euronext.csv")):
            tickers_PA = pd.read_csv(os.path.join(pf.get_path_app(), "Euronext.csv"))["Ticker"].values
            tickers_PA = [x for x in tickers_PA if x.endswith(".PA")]
            tickers = list(tickers) + list(tickers_PA)

        len_sp5, value = len(tickers), 0
        
        for tick in tickers:
            path_to_file = os.path.join(dataset_path, tick + ".parquet")
            data = yf.Ticker(tick).history(period="20y", interval='1d').reset_index()
            if not data.empty :
                if "Dividends" in data :
                    data["Dividends"] = data["Dividends"].astype(str).str.replace(r'[^0-9.]', '', regex=True)
                    data["Dividends"] = pd.to_numeric(data["Dividends"].values, errors="coerce")
                data.to_parquet(path_to_file, compression="brotli")

            my_bar.progress(value / len_sp5, f"SP500 {value}/{len_sp5} {tick}")
            my_toast.toast( f"Updated {tick}", icon=":material/check_small:", duration="infinite")
            value += 1
            time.sleep(time_wait[0])
            if value % 20 == 0 :
                time.sleep(time_wait[1])
            if value % 500 == 0 :
                time.sleep(time_wait[2])
            if value % 1000 == 0 :
                time.sleep(time_wait[3])





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
    my_toast.toast("Update done", icon=":material/check_small:", duration="short")
    my_bar.empty()
    st.badge("Database update done", color="green", icon=":material/check_small:")
