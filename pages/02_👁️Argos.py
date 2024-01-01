import streamlit as st
import pandas as pd
import numpy as np
import os
from app.shared import get_dataset_path, get_tickers_path, get_sqlite_word
import app.ta as ta #import df_check

st.set_page_config(layout = 'wide')

from app.logging import logging
logging(st.secrets["secret1"], st.secrets["secret2"])

st.caption("*NOT FINANCIAL ADVICE!! FOR EDUCATION ONLY*")

'# :eye: Argos'
st.caption("""_Argus Panoptes (Ἄργος Πανόπτης) was the guardian of the heifer-nymph Io and the son of Arestor. According to Asclepiades, Argus Panoptes was a son of Inachus, and according to Cercops he was a son of Argus and Ismene, daughter of Asopus. Acusilaus says that he was earth-born (authochthon), born from Gaia. Probably Mycene (in another version the son of Gaia) was a primordial giant whose epithet Panoptes, "all-seeing", led to his being described with multiple, often one hundred eyes._""")

col1, col2, col3 = st.columns(3)
market = col1.radio('Market', ['sp500', 'crypto'], horizontal=True)
if market == "crypto" :
    boc = col2.radio("Crypto broker", ["binance", "coinbase"], horizontal=True)

path = f"dataset/{market}"
if market == "crypto" :
    path = path + f"_{boc}"
path = path + "/"
'---'
col1, col2, col3, col4, col5 = st.columns(5)
ab_rsi, un_rsi = col1.checkbox('Above rsi'), col2.checkbox('Under rsi')
ab_rsi_number = col1.slider('rsi >', 0, 100, 90, 10) if ab_rsi else None
un_rsi_number = col2.slider('rsi <', 0, 100, 10, 10) if un_rsi else None

sqz, twz = col3.checkbox('Squeeze'), col4.checkbox('Tweezer')
twz_type = col4.radio('TWZ : bear or bear?', ['bearish', 'bullish'], horizontal=True) if twz else None

bbands = col1.checkbox('BB')
above_under_bb = col1.radio('above or under BB?', ['above', 'under'], horizontal=True) if bbands else None

um_ham, um_ham_mean = col2.checkbox('umbrella/hammer'), col3.checkbox('um_ham_mean')
umbrella_or_hammer = col2.radio('umbrella or hammer?', ['umbrella', 'hammer'], horizontal=True) if um_ham else None
um_ham_mean_kind = col3.radio('bear or bull?', ['bearish', 'bullish'], horizontal=True) if um_ham_mean else None

touching_ma20, divergence = col4.checkbox('touching SMA20'), col5.checkbox('divergences')
compression = col5.checkbox('Compression')
vlum = col1.checkbox("High volume")

''
go = st.button("Let's go!")
'---'

if go :
    tickers = [x.replace(".parquet", "") for x in os.listdir(path)]
    for symbol in tickers :
        df_check = ta.df_check()
        count = 0

        try :
            data = pd.read_parquet(f'{path}{symbol}.parquet')
        except :
            continue

        data['Open'] = data['Open'].astype(float)
        data['Close'] = data['Close'].astype(float)
        data['High'] = data['High'].astype(float)
        data['Low'] = data['Low'].astype(float)

        if len(data) > 0 :
            if ab_rsi :
                count+=1
                ab_rsi_result = ta.check_if_above_rsi(data, 14, ab_rsi_number)
                df_check.loc[df_check['ta_ref'] == 'above_rsi', 'result'] = str(ab_rsi_result)

            if un_rsi :
                count+=1
                un_rsi_result = ta.check_if_under_rsi(data, 14, un_rsi_number)
                df_check.loc[df_check['ta_ref'] == 'under_rsi', 'result'] = str(un_rsi_result)
            if sqz :
                count+=1
                sqz_result = ta.check_if_sqz(data, 20)
                df_check.loc[df_check['ta_ref'] == 'squeeze', 'result'] = sqz_result
            if bbands :
                count+=1
                if above_under_bb == 'above' :
                    ab_bb_results = ta.check_above_bband(data, 20)
                    df_check.loc[df_check['ta_ref'] == 'above_bb', 'result'] = ab_bb_results
                elif above_under_bb == 'under' :
                    un_bb_results = ta.check_under_bband(data, 20)
                    df_check.loc[df_check['ta_ref'] == 'under_bb', 'result'] = un_bb_results
            if twz :
                count+=1
                if twz_type == 'bearish' :
                    twz_result = ta.check_twz_bear(data, range_accept = 0.1/100, percent_retrace = 0.33/100)
                    df_check.loc[df_check['ta_ref'] == 'tweezer', 'result'] = twz_result
                elif twz_type == 'bullish' :
                    twz_result = ta.check_twz_bull(data, range_accept = 0.1/100, percent_retrace = 0.33/100)
                    df_check.loc[df_check['ta_ref'] == 'tweezer', 'result'] = twz_result

            if um_ham :
                count+=1
                if umbrella_or_hammer == 'umbrella' :
                    result_u = ta.check_U(data)
                    df_check.loc[df_check['ta_ref'] == 'um', 'result'] = result_u

                elif umbrella_or_hammer == 'hammer' :
                    result_h = ta.check_H(data)
                    df_check.loc[df_check['ta_ref'] == 'ham', 'result'] = result_h

            if um_ham_mean :
                count+=1
                if um_ham_mean_kind == 'bearish' :
                    result_hu_mean = ta.check_HU_mean(data, 24)
                    if "-" in result_hu_mean :
                        df_check.loc[df_check['ta_ref'] == 'hu_mean', 'result'] = 'bearish(' + result_hu_mean + ')'

                elif um_ham_mean_kind == 'bullish' :
                    result_hu_mean = ta.check_HU_mean(data, 24)
                    if ("-" not in result_hu_mean) & (result_hu_mean != "0") :
                        df_check.loc[df_check['ta_ref'] == 'hu_mean', 'result'] = 'bullish(' + result_hu_mean + ')'

            if divergence :
                count+=1
                data, data_bottom, data_top = ta.bull_bear_rsi_div(data)
                if (len(data_bottom) > 4) & (len(data_top) > 4) :
                    if (data_bottom.iloc[-2]['bullish_rsi_div'] == True) | (data_bottom.iloc[-3]['bullish_rsi_div'] == True) | (data_bottom.iloc[-4]['bullish_rsi_div'] == True) :
                        df_check.loc[df_check['ta_ref'] == 'divergence', 'result'] = 'bullish rsi 🐸'
                    if (data_top.iloc[-2]['bearish_rsi_div'] == True) | (data_top.iloc[-3]['bearish_rsi_div'] == True) | (data_top.iloc[-4]['bearish_rsi_div'] == True)  :
                        df_check.loc[df_check['ta_ref'] == 'divergence', 'result'] = 'bearish rsi 🐻'

            if compression :
                count+=1
                compr_detected = ta.compression(data)
                if compr_detected in ["rising compression 🐻", "falling compression 🐸"] :
                    df_check.loc[df_check['ta_ref'] == 'compression', 'result'] = compr_detected
                else :
                    df_check.loc[df_check['ta_ref'] == 'compression', 'result'] = False

            if vlum :
                count+=1
                data.Volume=data.Volume.astype(float)
                if (data['Volume'].quantile(q=0.8) < data['Volume'].iloc[-1])|(data['Volume'].quantile(q=0.8) < data['Volume'].iloc[-2]) :
                    df_check.loc[df_check['ta_ref'] == 'volume', 'result'] = 'high 🔥'


            # df_check
            # st.stop()

            df_check = df_check[(df_check['result'] != '') & (df_check['result'] != False) & (df_check['result'] != 'False') & (df_check['result'] != 'off')]
            if (len(df_check) == count) & (len(df_check) > 0) :
                if market == 'crypto' :
                    link_binance = f"[{symbol}(Binance)](https://www.tradingview.com/chart/?symbol=BINANCE:{symbol})"
                    st.write(link_binance + ' : ' + df_check.to_string(index=False).replace('ta_ref', '').replace('result', ''))

                else:
                    link_NASDAQ = f"[{symbol}(NASDAQ)](https://www.tradingview.com/chart/?symbol=NASDAQ:{symbol})"
                    link_NYSE = f"[{symbol}(NYSE)](https://www.tradingview.com/chart/?symbol=NYSE:{symbol})"
                    st.write(link_NASDAQ + '/' + link_NYSE + ' : ' + df_check.to_string(index=False).replace('ta_ref', '').replace('result', ''))
    st.success('...Done')
