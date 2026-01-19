import streamlit as st
import pandas as pd
import numpy as np
import os
import functions.ta as ta
import plotly.express as px
import plotly.graph_objects as pgo
import functions.pathfunc as pf

from functions.logging import logging

st.set_page_config(layout = 'wide')


logging(st.secrets["secret1"], st.secrets["secret2"])
"# :material/ophthalmology: Argos"
st.caption("""_Argus Panoptes (Ἄργος Πανόπτης) was the guardian of the heifer-nymph Io and the son of Arestor. According to Asclepiades, Argus Panoptes was a son of Inachus, and according to Cercops he was a son of Argus and Ismene, daughter of Asopus. Acusilaus says that he was earth-born (authochthon), born from Gaia. Probably Mycene (in another version the son of Gaia) was a primordial giant whose epithet Panoptes, "all-seeing", led to his being described with multiple, often one hundred eyes._""")

col1, col2, col3 = st.columns(3)
market = col1.radio('Market', ['stocks', 'crypto'], horizontal=True)
if market == "stocks" :
    list_submarkets = sorted([ x.replace(".csv","") for x in os.listdir(os.path.join(pf.get_path_data(), "tickers list")) ])
    submarkets = st.pills("Sub market", list_submarkets, selection_mode = "multi")
    disable_search = False
    if len(submarkets) == 0 :
        disable_search = True

    tickers = []
    for sm in submarkets :
        tickers.extend(os.listdir(os.path.join(pf.get_path_data(), "stocks", sm )))

    if len(tickers)>0 :
        st.caption(f":blue-badge[{len(tickers)}] stocks are going to be screened.")


elif market == "crypto" :
    boc = col2.radio("Crypto broker", ["binance"], horizontal=True)
    disable_search = False
    submarkets=["crypto_binance"]


'---'

c1, c2, c3 = st.columns(3)
with c1.expander("RSI", expanded=True) :
    cl1, cl2 = st.columns(2)
    ab_rsi, un_rsi = cl1.checkbox('Above rsi'), cl2.checkbox('Under rsi')
    ab_rsi_number = st.slider('above rsi', 0, 100, 90, 10, help="Looking for all asset with RSI above this value") if ab_rsi else None
    un_rsi_number = st.slider('under rsi', 0, 100, 10, 10, help="Looking for all asset with RSI under this value") if un_rsi else None

with c2.expander("Candlesticks", expanded=True) :
    cl1, cl2 = st.columns(2)
    twz = cl1.checkbox('Tweezer', help="Warning, code to update")
    um_ham = cl2.checkbox('umbrella/hammer')
    umbrella_or_hammer = cl2.radio('umbrella or hammer?', ['umbrella', 'hammer'], horizontal=True) if um_ham else None
    twz_type = cl1.radio('TWZ : bear or bear?', ['bearish', 'bullish'], horizontal=True) if twz else None

with c3.expander("Bollinger bands", expanded=True) :
    bbands = st.checkbox('Bollinger bands')
    above_under_bb = st.radio('above or under BB?', ['above', 'under'], horizontal=True, help="Checking if 'High' above BB or 'Low' under BB") if bbands else None


with c1.expander("Umbrella hammer count", expanded=True) :
    um_ham_mean = st.checkbox('umb/ham count')
    um_ham_mean_kind = st.radio('bear or bull?', ['bearish', 'bullish'], horizontal=True) if um_ham_mean else None

with c2.expander("Squeeze, volume & wick", expanded=True) :
    sqz = st.checkbox('Squeeze', help="Checking if BB in between KC")
    wick = st.checkbox("Wick trend", help="high wicks vs low wicks")
    if wick :
        col1, col2 = st.columns(2)
        wick_trend = col1.radio("Wick trend", ["Bullish", "Bearish"], help="Checks if last candle is bull or bear")
        wick_breakout = col2.toggle("Wick breakout", help="Last candle flipped to bull or bear")
    vlum = st.checkbox("High volume", help="Are your assets under the radar")
    if vlum :
        perc=st.slider("Above percentile", 0., 1., .85, help="Last volume value above this percentile")
        macro_vlum = st.toggle("Macro", help="50SMA volume")

    DOT=st.checkbox("Dots trend streategy", help="mixing AO and RSI to evaluate trends")
    if DOT :
        col1, col2 = st.columns(2)
        dot_trend = col1.radio("Type of dot trend", ["Bullish", "Bearish"], help="Checks if last candle turned bull to bear or bear to bull")
        dot_breakout = col2.toggle("Dot breakout", help="Last candle flipped to bull or bear")



with c3.expander("TMA20, Div, Comp", expanded=True) :
    touching_ma20, divergence = st.checkbox('touching SMA20'), st.checkbox('divergences')
    compression = st.checkbox('Compression')

weekly = st.toggle("weekly")

''
c1,c2 = st.columns([1,5])
go = c1.button(":material/search: Screen the market", type="primary", disabled=disable_search)


'---'

if go :
    toast_tickers = st.toast("Screening...", duration="infinite", icon=":material/search:")
    toast_found = st.toast("No match yet...", duration="infinite")

    for sm in submarkets :
        dataset_path = os.path.join(pf.get_path_data(), sm) if market == "crypto" else os.path.join(pf.get_path_data(), "stocks", sm)
        f"### {sm}"
        tickers = [x.replace(".parquet", "") for x in os.listdir(dataset_path)]

        for symbol in tickers :
            toast_tickers.toast(f"Screening... {symbol}", duration="infinite", icon=":material/search:")
            df_check = ta.df_check()
            count = 0

            try :
                data = pd.read_parquet(os.path.join(dataset_path,f"{symbol}.parquet"))
            except :
                continue

            for elem in ["Open", "Close", "High", "Low", "Volume"] :
                data[elem] = data[elem].astype(float)

            if weekly :
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

            if not len(data) > 20 :
                continue

            if len(data) > 20 :
                if wick :
                    count+=1
                    if len(data) > 3 :
                        data["wick_trend"]=None
                        data["wick"]=(data["Close"].values-data["Low"].values) - (data["High"].values-data["Close"].values)
                        data["wick"] = data["wick"].ewm(span=20, adjust=False).mean()
                        data.loc[data["wick"].values > 0, "wick_trend"] = "Bullish"
                        data.loc[data["wick"].values < 0, "wick_trend"] = "Bearish"
                        wt = data["wick_trend"].values
                        cond = ((wt[-2] != wick_trend) & (wt[-1] == wick_trend)|(wt[-3] != wick_trend) & (wt[-2] == wick_trend)) if wick_breakout else ((wt[-1] == wick_trend) | (wt[-2] == wick_trend))
                        if cond :
                            df_check.loc[df_check['ta_ref'] == 'wick', 'result'] = wick_trend


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
                    try:
                        data, data_bottom, data_top = ta.bull_bear_rsi_div(data)
                    except:
                        continue
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
                    if len(data) > 3 :
                        if macro_vlum :
                            macro_vol = data["Volume"].rolling(50).mean()
                            if (macro_vol.quantile(q=perc) < macro_vol.iloc[-1])|(macro_vol.quantile(q=perc) < macro_vol.iloc[-2]) :
                                df_check.loc[df_check['ta_ref'] == 'volume', 'result'] = 'high 🔥'
                        else :
                            if (data['Volume'].quantile(q=perc) < data['Volume'].iloc[-1])|(data['Volume'].quantile(q=perc) < data['Volume'].iloc[-2]) :
                                df_check.loc[df_check['ta_ref'] == 'volume', 'result'] = 'high 🔥'

                if DOT :
                    count+=1
                    # Calculate Dot
                    # Calculate "dot" and "trendline" indicators
                    if len(data) > 3 :
                        data["dot"] = data["Close"].ewm(span=20, adjust=False).mean()
                        data["trendline"] = data["Close"].ewm(span=20, adjust=False).mean().ewm(span=20, adjust=False).mean()
                        if "ao" not in data :
                            data['ao'] = ta.ao(data)
                            data['bob_ao'] = ta.bob_ao(data)

                        # Determine trend based on "dot" and "trendline" indicators
                        data.loc[data["dot"] > data["trendline"], 'sentiment'] = 'Bullish'
                        data.loc[data["dot"] < data["trendline"], 'sentiment'] = 'Bearish'
                        if "RSI14" not in data :
                            data["RSI14"] = ta.RSI(data, 14)

                        data.loc[ (data["RSI14"] > 40) & (data["RSI14"] < 60) , "sentiment"] = ""
                        dot_col = data["sentiment"].values
                        cond = ((dot_col[-1] == dot_trend)&(dot_col[-2] != dot_trend)) | ((dot_col[-2] == dot_trend)&(dot_col[-3] != dot_trend)) if dot_breakout else ((dot_col[-1] == dot_trend)|(dot_col[-2] == dot_trend))
                        if cond :
                            df_check.loc[df_check['ta_ref'] == 'dot', 'result'] = dot_trend





                # df_check
                # st.stop()
                df_check = df_check[(df_check['result'] != '') & (df_check['result'] != False) & (df_check['result'] != 'False') & (df_check['result'] != 'off')]

                if (len(df_check) == count) & (len(df_check) > 0) :
                    toast_found.toast(f"Found {symbol}", duration="infinite", icon=":material/check_small:")
                    text_str = df_check.to_string(index=False).replace('ta_ref', '').replace('result', '')


                    @st.fragment
                    def result_argos(symbol, market, sm, weekly, text_str) :
                        c1,c2 = st.columns([1,5])
                        if market == 'crypto' :
                            link = f"[{symbol}(Binance)](https://www.tradingview.com/chart/?symbol=BINANCE:{symbol})"
                            c1.write(link + ' : ' + text_str)
                        else :
                            links = [f"[{symbol}(NASDAQ)](https://www.tradingview.com/chart/?symbol=NASDAQ:{symbol})" , f"[{symbol}(NYSE)](https://www.tradingview.com/chart/?symbol=NYSE:{symbol})"]
                            c1.write(links[0] + '/' + links[1] + ' : ' + text_str)

                        if c2.button(f"Show chart {symbol}") :
                            ds_path = os.path.join(pf.get_path_data(), sm) if market == "crypto" else os.path.join(pf.get_path_data(), "stocks", sm)
                            dataframe = pd.read_parquet(os.path.join(ds_path, f"{symbol}.parquet"))



                            @st.dialog(symbol, width="large")
                            def argos_dialog(dataframe) :
                                if st.toggle("Volume markers") :
                                    mrk_max_size = st.slider("Marker max size", 5, 50, 20)
                                    fig = pgo.Figure()

                                    # st.write( dataframe )
                                    dataframe["Volume"] = pd.to_numeric(dataframe["Volume"], errors="coerce")
                                    dataframe_vol = dataframe[dataframe["Volume"]>0]

                                    if weekly :
                                        for elem in ["Open", "Close", "High", "Low"] :
                                            dataframe[elem] = dataframe[elem].astype(float)

                                        dataframe['week'] = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y').dt.isocalendar().week.astype(str)
                                        dataframe["week"] = [ w if len(w)>1 else "0"+w for w in dataframe["week"].values ]
                                        dataframe['year'] = pd.to_datetime(dataframe['Date'], format='%d/%m/%Y').dt.isocalendar().year.astype(str)
                                        dataframe["week"] = [ y + "_" + w for w,y in zip(dataframe["week"].values, dataframe["year"].values)]

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

                                        dataframe = dataframe.groupby("week").agg(aggregation).reset_index()
                                        dataframe = dataframe.sort_values(by="week").reset_index(drop=True)

                                    fig.add_trace(pgo.Scatter(x=dataframe_vol["Date"].values, y=dataframe_vol["Close"].values, name="Volume markers", mode="markers",
                                                             marker_color=dataframe_vol["Volume"].values, marker_size=dataframe_vol["Volume"].values/max(dataframe_vol["Volume"].values)*mrk_max_size,
                                                             marker_colorscale="Jet"))


                                    # fig = px.scatter(dataframe, x="Date", y="Close", size="Volume", color="Volume", opacity="Opacity", color_continuous_scale="Jet")
                                    fig.update_traces(marker_line_width=0)
                                    fig.update_layout(
                                        xaxis=dict(title='Date'),  # Nom de l'axe des x
                                        yaxis=dict(title='Close'),  # Nom de l'axe des y
                                        title=f'{symbol} preview'  # Titre du graphique
                                    )
                                else :
                                    fig = px.line(dataframe, x="Date", y="Close")
                                    fig.update_traces(line_width=1)
                                    fig.update_layout(title=f'{symbol} preview')
                                st.plotly_chart(fig)
                            argos_dialog(dataframe)
                    result_argos(symbol, market, sm, weekly, text_str)





    st.badge('Done', color="green", icon=":material/check_small:")
    toast_found.toast("Done", duration="short")
    toast_tickers.toast("Screening complete", icon=":material/check_small:", duration="short")

    st.caption("*NOT FINANCIAL ADVICE!! FOR EDUCATION ONLY*")
