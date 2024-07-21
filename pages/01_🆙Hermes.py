import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
import datetime
import os
from app import HistoricalData, Cryptocurrencies, LiveCryptoData
from app.shared import get_tickers_path, get_dataset_path

from app.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])

"🆙"
st.caption("""  _Hermes (/ˈhɜːrmiːz/; Greek: Ἑρμῆς) is an Olympian deity in ancient Greek religion and mythology considered the herald of the gods.
He is also considered the protector of human heralds, travelers, thieves, merchants, and orators. He is able to move quickly and freely between the worlds of the mortal and the divine aided by his winged sandals. Hermes plays the role of the psychopomp or "soul guide"—a conductor of souls into the afterlife.
In myth, Hermes functions as the emissary and messenger of the gods, and is often presented as the son of Zeus and Maia, the Pleiad. He is regarded as "the divine trickster", about which the Homeric Hymn to Hermes offers the most well-known account._
""")
st.caption("Better to update crypto database at 12:00 UTC (=daily candle close). Binance=Binance US compatible coins. Update speed : Binance=fast, SP500+Euronext=normal, Coinbase=slow initialisation then normal.")

import requests
import ast

#binance
def import_crypto(ticker, tframe, start_date=None) :
    apis = ["https://api.binance.com", "https://api-gcp.binance.com", "https://api1.binance.com",
            "https://api.binance.us", "https://api2.binance.com", "https://api3.binance.com",
            "https://api4.binance.com"]

    url = f"{apis[3]}/api/v3/klines?symbol={ticker}&interval={tframe}"
    payload={}
    headers = {}
    response = requests.request("GET", url, headers=headers, data=payload)
    data = ast.literal_eval(response.text)

    df = pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume", "Timestamp_end", "Amount", "Count", "Open_interest", "Turnover", "Taker_buy_base_asset_volume"])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['order'] = df['Date'].dt.strftime('%Y%m%d')
    df['order'] = df['order'].astype(int)

    df['Date'] = df['Date'].dt.strftime('%d/%m/%Y')
    df['Date'] = df['Date'].astype(str).str[:9]
    df['timeframe'] = tframe
    df = df.sort_values(by=['order'])
    return df

#coinbase
def update_pair(pair, date, pair_path) :
    new = HistoricalData(pair,86400,date).retrieve_data()
    if not new.empty :
        cols = new.columns
        new_cols=[]
        for i in range(len(new.columns)) :
            new_cols.append(cols[i].title())
        new.columns = new_cols
    
        new["timeframe"] = "1d"
        new=new.reset_index()
        new["yr"],new["mth"],new["dy"] = new["time"].dt.year, new["time"].dt.month, new["time"].dt.day
        new["Day"], new["Month"] = new["dy"].astype(str), new["mth"].astype(str)
        new.loc[new["dy"]<10, "Day"] = "0"+new["Day"]
        new.loc[new["mth"]<10, "Month"] = "0"+new["Month"]
    
        new["Date"] = new["Day"] + "/" + new["Month"] +"/"+ new["yr"].astype(str)
        new["pair"] = pair
        return new.drop(columns=["dy", "yr", "mth", "Day", "Month"])
    else :
        return new



col1,col2,col3 = st.columns(3)
market = col1.radio('Market to update', ['sp500', 'crypto'])
tickers_sp500_path, tickers_etoro_path, tickers_binance_path = get_tickers_path()
if market == 'sp500' :
    market_tickers_path = tickers_sp500_path
    boc="binance"
elif market == 'crypto' :
    boc=col2.radio("Crypto broker to update", ["binance", "coinbase"])
    market_tickers_path = tickers_binance_path



dataset_path="dataset/sp500/" if market=="sp500" else f"dataset/crypto_{boc}/"

# dataset_path = get_dataset_path(market)

today_utc = datetime.datetime.utcnow().strftime('%Y-%m-%d')

update = st.button('Update !', type="primary")
st.caption(f"Today's UTC date : {today_utc}")

if update :
    if ((market == "crypto") & (boc == "binance")) :
        if os.path.exists(dataset_path) :
            list_files = os.listdir(dataset_path)
            for file in list_files :
                os.remove(dataset_path+file)

    if market == 'sp500' :
        my_bar = st.progress(0., "SP500 updating datas")
        tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        tickers=tickers['Symbol'].values
        if os.path.exists("PA_tickers/Euronext.csv") :
            tickers_PA=pd.read_csv("PA_tickers/Euronext.csv")["Ticker"].values
            tickers_PA=[x for x in tickers_PA if x.endswith(".PA")]
            tickers = list(tickers) + list(tickers_PA)
        
        col1, col2, col3 = st.columns(3)
        #read lines from csv tickers

        len_sp5, value = len(tickers), 0
        for tick in tickers :
            path_to_file = dataset_path+tick+".parquet"
            if os.path.exists(path_to_file) :
                data = pd.read_parquet(path_to_file)
                if len(data) > 10 :
                    to_drop=9
                    from_date = data["Date"].astype(str).values[-to_drop][:10]
                    new_data = yf.download(tickers = tick, start = from_date, interval = "1d").reset_index()
                    data.drop(data.tail(to_drop).index, inplace=True)
                    data=pd.concat([data,new_data], ignore_index=True)
                    data.to_parquet(path_to_file, compression="brotli")

                else :
                    data = yf.Ticker(tick).history(period="max", interval='1d').reset_index()
                    data.to_parquet(path_to_file, compression="brotli")
            else:
                data = yf.Ticker(tick).history(period="max", interval='1d').reset_index()
                data.to_parquet(path_to_file, compression="brotli")

            my_bar.progress(value/len_sp5, f"SP500 {value}/{len_sp5} {tick}")
            value+=1

        # with open(market_tickers_path) as f_sp500 :
        #     lines_sp500 = f_sp500.read().splitlines()
        #     #use tickers to update db
        #     len_sp5, value = len(lines_sp500), 0
        #     for symbol_sp500 in lines_sp500 :
        #         data_sp500 = yf.Ticker(symbol_sp500).history(period="max", interval='1d').reset_index()
        #         data_sp500.to_parquet(dataset_path + symbol_sp500 + ".parquet", compression="brotli")
        #         my_bar.progress(value/len_sp5, f"SP500 {value}/{len_sp5} {symbol_sp500}")
        #         value+=1
        #         # symbol_sp500.replace(".","_"), ' created...'

    elif market == 'crypto' :
        if boc == "binance" :
            import json
            my_bar = st.progress(0., "Binance updating datas")
            tickers_data=requests.request("GET", "https://api.binance.us/api/v3/exchangeInfo")
            tickers_data=json.loads(tickers_data.content)
            tickers_data = tickers_data["symbols"]
            tickers = [x["symbol"] for x in tickers_data]
            status = [x["status"] for x in tickers_data]
            len_tick = len([x for x,y in zip(tickers,status) if y == "TRADING"])
            value=1
            for tick,stat in zip(tickers,status) :
                path_to_file=dataset_path+tick+".parquet"
                if stat != "TRADING" :
                    if os.path.exists(path_to_file) :
                        os.remove(path_to_file)
                        continue
                if stat == "TRADING" :
                    data = import_crypto(tick, '1d')
                    if len(data)>0 :
                        data.to_parquet(path_to_file, compression="brotli")
                    my_bar.progress(value/len_tick, f"Binance {value}/{len_tick} {tick}")
                    value+=1


            # st.stop()
            # my_bar = st.progress(0., "Binance updating datas")
            # tickers = pd.read_csv('tickers/tickers_binance.csv')
            # tickers = tickers['tickers'].unique()
            # col1, col2, col3 = st.columns(3)
            # #read lines from csv tickers
            # #use tickers to update db
            # len_tick = len(tickers)
            # value = 1
            # for symbol in tickers :
            #     try :
            #         data = import_crypto(symbol, '1d')
            #         data.to_parquet(dataset_path+symbol+".parquet", compression="brotli")
            #     except :
            #         st.error('Failed update for ' + symbol + ' on second path')
            #
            #     my_bar.progress(value/len_tick, f"Binance {value}/{len_tick} {symbol}")
            #     value+=1

        elif boc == "coinbase" :
            my_bar = st.progress(0., "Coinbase updating datas")
            path="dataset/crypto_coinbase/"
            coins = Cryptocurrencies().find_crypto_pairs()
            pairs_to_remove = coins[coins.status != "online"].id.unique()
            for pair_r in pairs_to_remove :
                if os.path.exists(path+pair_r) :
                    os.remove(path+pair_r)
            coins = coins[coins.status=="online"]
            pairs = coins.id.unique()
            # pairs = [x for x in pairs if (x[-3:]=="USD")|(x[-3:]=="BTC")]

            len_pairs = len(pairs)
            value = 1
            for pair in pairs :
                pair_path = path + pair + ".parquet"
                start_date = "2017-11-01-00-00"
                if os.path.exists(pair_path) :
                    data = pd.read_parquet(pair_path)
                    if len(data) > 2 :
                        entry_date = data["Date"].values[-2]
                        entry = entry_date.split("/")
                        if len(entry)>0 :
                            start_date = f"{entry[2]}-{entry[1]}-{entry[0]}-00-00"
                            new = update_pair(pair, start_date, pair_path)
                            if len(new) > 0 :
                                data.drop(data.tail(2).index,inplace=True)
                                data = pd.concat([data, new], ignore_index=True)
                            else :
                                continue
                        else :
                            data = update_pair(pair, start_date, pair_path)
                    else :
                        data = update_pair(pair, start_date, pair_path)
                else :
                    data = update_pair(pair, start_date, pair_path)

                if not data.empty :
                    data.to_parquet(pair_path, compression="brotli")
                my_bar.progress(value/len_pairs, f"Coinbase {value}/{len_pairs} {pair}")
                value+=1
    my_bar.empty()
    col1.success(market + ' updated !')

with st.sidebar.popover("Dev update tool", use_container_width=True) :
    if st.toggle('Download zipped .parquets', help = 'download zip of all the .parquets available in the cloud. Useful to upload it on a gdrive') :
        list_paths = [os.path.join("dataset", x) for x in ["sp500","crypto_coinbase", "crypto_binance"] ]
        
        def convert_df(df):
            return df.to_parquet()
        
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "x") as parquet_zip:
            for x in list_paths :
                st.toast(f"Processing {x}")
                if not os.path.exists(x) :
                    st.toast(f"{x} does not exist")
                    continue
                list_files = [y for y in os.listdir(x) if y.endswith(".parquet")]
                my_bar = st.progress(0., x)
                len_list_files = len(list_files)
                for i,f in enumerate(list_files) :
                    my_bar.progress((i+1)/len_list_files, os.path.join(x,f))
                    # parquet_zip.write(f"{f}_{x.split('_')[-1].replace('/','')}", os.path.join(x,f))
                    parquet_zip.writestr(f"{f}_{x.split('_')[-1].replace('/','')}", convert_df(pd.read_parquet(os.path.join(x,f))) )
                my_bar.empty()
    
        st.download_button(label="Download .zip", data=buf.getvalue(), file_name="historical_data.zip", mime="application/zip", help="upload the zipfile to g drive")
