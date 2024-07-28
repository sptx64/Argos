import streamlit as st
import os
import pandas as pd
from app.ta import ao, bob_ao, get_squeeze, get_kc, HU, RSI


'# Asset Rotation Aperture'
D1='Feature Switch A'

res_htf = 60
alt_vol_src = 'INDEX:BTCUSD'
res = 15
oscMode=D1
oscLength=3600
smoothing=360
postFilter=True
opacity_A=100
opacity_B=14

st.caption("NOT FINANCIAL ADVICE!! FOR EDUCATION ONLY")

c1,c2=st.sidebar.columns(2)

market = c1.radio('Market', ['sp500', 'crypto'], index=1)

broker="binance"
if market == "crypto" :
    broker = c2.radio("broker", ["binance","coinbase"], index=1)

path = f'dataset/{market}_{broker}/' if market == "crypto" else f'dataset/{market}/'
tables = [x.replace(".parquet","") for x in os.listdir(path)]

# Create dropdown menu to select ticker
val=None
if market == "crypto" :
    if broker == "binance" :
        val=["BTCUSDT", "ETHUSDT", "DOGEUSDT", "AVAXUSDT", "FTMUSDT", "SOLUSDT", "ATOMUSDT"]
    elif broker == "coinbase":
        val=["BTC-USDT", "ETH-USDT", "DOGE-USDT", "AVAX-USDT", "RONIN-USDT", "SOL-USDT", "ATOM-USDT"]
    else :
        val=None

ticker = st.sidebar.multiselect("Select a ticker:", tables, val)

dfs={}
for t in ticker :
    file_path = os.path.join(path, f"{t}.parquet")
    file_path
    dfs[t] = pd.read_parquet(file_path)
    list(dfs[t]),
    dfs[t],
    dfs[t]["RSI"] = RSI(dfs[t], 14)
    
    

    







# mode_selector (src, len, sel, filt) =>
#     selector    =  switch sel
#         D01     => ta.cmo(ta.mfi(src, len), (len/2))
#     filt        ?  ta.vwma (selector, smoothing)  :  ta.ema (selector, 6)

# // SOURCES HTF
# [source_4h_1, asset_4h_1] = request.security(symbol_1, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_2, asset_4h_2] = request.security(symbol_2, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_3, asset_4h_3] = request.security(symbol_3, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_4, asset_4h_4] = request.security(symbol_4, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_5, asset_4h_5] = request.security(symbol_5, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_6, asset_4h_6] = request.security(symbol_6, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_7, asset_4h_7] = request.security(symbol_7, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_8, asset_4h_8] = request.security(symbol_8, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)

# // Oscilator with Current Symbol Volume Calc
# asset_1 = request.security(symbol_1, res, mode_selector(source_4h_1, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_2 = request.security(symbol_2, res, mode_selector(source_4h_2, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_3 = request.security(symbol_3, res, mode_selector(source_4h_3, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_4 = request.security(symbol_4, res, mode_selector(source_4h_4, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_5 = request.security(symbol_5, res, mode_selector(source_4h_5, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_6 = request.security(symbol_6, res, mode_selector(source_4h_6, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_7 = request.security(symbol_7, res, mode_selector(source_4h_7, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_8 = request.security(symbol_8, res, mode_selector(source_4h_8, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)

# // Oscilator with Volume Symbol Volume Calc
# asset_1 := use_asset_1 ? not na(asset_1) ? asset_1 : request.security( alt_vol_src, res, mode_selector(source_4h_1, oscLength, oscMode, postFilter) ) : na
# asset_2 := use_asset_2 ? not na(asset_2) ? asset_2 : request.security( alt_vol_src, res, mode_selector(source_4h_2, oscLength, oscMode, postFilter) ) : na
# asset_3 := use_asset_3 ? not na(asset_3) ? asset_3 : request.security( alt_vol_src, res, mode_selector(source_4h_3, oscLength, oscMode, postFilter) ) : na
# asset_4 := use_asset_4 ? not na(asset_4) ? asset_4 : request.security( alt_vol_src, res, mode_selector(source_4h_4, oscLength, oscMode, postFilter) ) : na
# asset_5 := use_asset_5 ? not na(asset_5) ? asset_5 : request.security( alt_vol_src, res, mode_selector(source_4h_5, oscLength, oscMode, postFilter) ) : na
# asset_6 := use_asset_6 ? not na(asset_6) ? asset_6 : request.security( alt_vol_src, res, mode_selector(source_4h_6, oscLength, oscMode, postFilter) ) : na
# asset_7 := use_asset_7 ? not na(asset_7) ? asset_7 : request.security( alt_vol_src, res, mode_selector(source_4h_7, oscLength, oscMode, postFilter) ) : na
# asset_8 := use_asset_8 ? not na(asset_8) ? asset_8 : request.security( alt_vol_src, res, mode_selector(source_4h_8, oscLength, oscMode, postFilter) ) : na

# // PLOTS
# zero_plot = plot (0, editable=false, display=display.none)
# plot_1  = plot (asset_1, 'Plot Asset 1',color.new (color_1, 100 - opacity_A), 1)
# fill_1  = plot (asset_1, 'Fill Asset 1',color.new (color_1, 100 - opacity_B), 0, plot.style_area)
# plot_2  = plot (asset_2, 'Plot Asset 2',color.new (color_2, 100 - opacity_A), 1)
# fill_2  = plot (asset_2, 'Fill Asset 2',color.new (color_2, 100 - opacity_B), 0, plot.style_area)
# plot_3  = plot (asset_3, 'Plot Asset 3',color.new (color_3, 100 - opacity_A), 1)
# fill_3  = plot (asset_3, 'Fill Asset 3',color.new (color_3, 100 - opacity_B), 0, plot.style_area)
# plot_4  = plot (asset_4, 'Plot Asset 4',color.new (color_4, 100 - opacity_A), 1)
# fill_4  = plot (asset_4, 'Fill Asset 4',color.new (color_4, 100 - opacity_B), 0, plot.style_area)
# plot_5  = plot (asset_5, 'Plot Asset 5',color.new (color_5, 100 - opacity_A), 1)
# fill_5  = plot (asset_5, 'Fill Asset 5',color.new (color_5, 100 - opacity_B), 0, plot.style_area)
# plot_6  = plot (asset_6, 'Plot Asset 6',color.new (color_6, 100 - opacity_A), 1)
# fill_6  = plot (asset_6, 'Fill Asset 6',color.new (color_6, 100 - opacity_B), 0, plot.style_area)
# plot_7  = plot (asset_7, 'Plot Asset 7',color.new (color_7, 100 - opacity_A), 1)
# fill_7  = plot (asset_7, 'Fill Asset 7',color.new (color_7, 100 - opacity_B), 0, plot.style_area)
# plot_8  = plot (asset_8, 'Plot Asset 8',color.new (color_8, 100 - opacity_A), 1)
# fill_8  = plot (asset_8, 'Fill Asset 8',color.new (color_8, 100 - opacity_B), 0, plot.style_area)

# // LABELS
# var label label_1  = use_asset_1 ? label.new(bar_index+8, asset_1, desc_1, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_1, size=size.small) : na
# var label label_2  = use_asset_2 ? label.new(bar_index+8, asset_2, desc_2, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_2, size=size.small) : na
# var label label_3  = use_asset_3 ? label.new(bar_index+8, asset_3, desc_3, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_3, size=size.small) : na
# var label label_4  = use_asset_4 ? label.new(bar_index+8, asset_4, desc_4, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_4, size=size.small) : na
# var label label_5  = use_asset_5 ? label.new(bar_index+8, asset_5, desc_5, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_5, size=size.small) : na
# var label label_6  = use_asset_6 ? label.new(bar_index+8, asset_6, desc_6, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_6, size=size.small) : na
# var label label_7  = use_asset_7 ? label.new(bar_index+8, asset_7, desc_7, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_7, size=size.small) : na
# var label label_8  = use_asset_8 ? label.new(bar_index+8, asset_8, desc_8, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_8, size=size.small) : na

# label.set_xy(label_1, bar_index+4, asset_1)
# label.set_xy(label_2, bar_index+4, asset_2)
# label.set_xy(label_3, bar_index+4, asset_3)
# label.set_xy(label_4, bar_index+4, asset_4)
# label.set_xy(label_5, bar_index+4, asset_5)
# label.set_xy(label_6, bar_index+4, asset_6)
# label.set_xy(label_7, bar_index+4, asset_7)
# label.set_xy(label_8, bar_index+4, asset_8)

# trading view pine source code
# // This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/

# // © ColeGarner

# //@version=5
# indicator (title = 'Asset Rotation Aperture', shorttitle = 'Asset Rotation Aperture', format=format.percent, precision=3, scale=scale.right)
# // v1.0
# string  D01    =  'Feature Switch A'

# //INPUTS

# res_htf     =                   '60'// 'Timeframe HTF Source', group="Hardcoded Soon")
# alt_vol_src =                   'INDEX:BTCUSD'//('INDEX:BTCUSD', 'Alt Volume source', group="Hardcoded Soon")
# res        =                    '15'//('15', 'Timeframe')
# oscMode    =                    D01 // 'Oscillator Mode', [D01, D02, D03])
# oscLength  =                    3600 //(3600, 'Length', minval=2)
# smoothing  =                    360// "Smoothing", minval = 1)
# postFilter =                    true
# opacity_A  =                    100
# opacity_B  =                    14
# use_asset_1=  input.bool        (true, '', inline='1', group='Main')
# color_1    =  input.color       (#ffa600, '', inline='1', group='Main')
# desc_1     =  input.string      ('BITCOIN', '', inline='1', group='Main')
# symbol_1   =  input.symbol      ('BINANCE:BTCUSDT', '', inline='1', group='Main')
# use_asset_2=  input.bool        (true, '', inline='2', group='Main')
# color_2    =  input.color       (#59e3ff, '', inline='2', group='Main')
# desc_2     =  input.string      ('ETHEREUM', '', inline='2', group='Main')
# symbol_2   =  input.symbol      ('BINANCE:ETHUSDT', '', inline='2', group='Main')
# use_asset_3=  input.bool        (true, '', inline='3', group='Main')
# color_3   =   input.color        (#fff200, '', inline='3', group='Main')
# desc_3    =   input.string       ('BINANCE COIN', '', inline='3', group='Main')
# symbol_3  =   input.symbol       ('BINANCE:BNBUSDT', '', inline='3', group='Main')
# use_asset_4=  input.bool         (true, '', inline='4', group='Main')
# color_4   =   input.color        (#ffffff, '', inline='4', group='Main')
# desc_4    =   input.string       ('DOGE', '', inline='4', group='Main')
# symbol_4  =   input.symbol       ('BINANCE:DOGEUSDT', '', inline='4', group='Main')
# use_asset_5=  input.bool         (true, '', inline='5', group='Main')
# color_5   =   input.color        (#2aff31, '', inline='5', group='Main')
# desc_5    =   input.string       ('SOL', '', inline='5', group='Main')
# symbol_5  =   input.symbol       ('BINANCE:SOLUSDT', '', inline='5', group='Main')
# use_asset_6=  input.bool         (true, '', inline='6', group='Main')
# color_6   =   input.color        (#9f90ff,'', inline='6', group='Main')
# desc_6    =   input.string       ('LINK', '',  inline='6', group='Main')
# symbol_6  =   input.symbol       ('BINANCE:LINKUSDT', '', inline='6', group='Main')
# use_asset_7=  input.bool         (true, '',inline='7', group='Main')
# color_7   =   input.color        (#ff5d5d, '', inline='7', group='Main')
# desc_7    =   input.string       ('UNI','', inline='7',group='Main')
# symbol_7  =   input.symbol       ('BINANCE:UNIUSDT', '',inline='7',group='Main')
# use_asset_8=  input.bool         (true, '', inline='8', group='Main')
# color_8   =   input.color        (#a7ffb8, '', inline='8',group='Main')
# desc_8    =   input.string       ('RUNE','', inline='8', group='Main')
# symbol_8  =   input.symbol       ('BINANCE:RUNEUSDT', '', inline='8', group='Main')




# // PROCESS --
# mode_selector (src, len, sel, filt) =>
#     selector    =  switch sel
#         D01     => ta.cmo(ta.mfi(src, len), (len/2))
#     filt        ?  ta.vwma (selector, smoothing)  :  ta.ema (selector, 6)

# // SOURCES HTF
# [source_4h_1, asset_4h_1] = request.security(symbol_1, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_2, asset_4h_2] = request.security(symbol_2, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_3, asset_4h_3] = request.security(symbol_3, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_4, asset_4h_4] = request.security(symbol_4, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_5, asset_4h_5] = request.security(symbol_5, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_6, asset_4h_6] = request.security(symbol_6, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_7, asset_4h_7] = request.security(symbol_7, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)
# [source_4h_8, asset_4h_8] = request.security(symbol_8, res_htf, [close, mode_selector(close, oscLength, oscMode, postFilter)], ignore_invalid_symbol=true)

# // Oscilator with Current Symbol Volume Calc
# asset_1 = request.security(symbol_1, res, mode_selector(source_4h_1, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_2 = request.security(symbol_2, res, mode_selector(source_4h_2, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_3 = request.security(symbol_3, res, mode_selector(source_4h_3, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_4 = request.security(symbol_4, res, mode_selector(source_4h_4, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_5 = request.security(symbol_5, res, mode_selector(source_4h_5, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_6 = request.security(symbol_6, res, mode_selector(source_4h_6, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_7 = request.security(symbol_7, res, mode_selector(source_4h_7, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)
# asset_8 = request.security(symbol_8, res, mode_selector(source_4h_8, oscLength, oscMode, postFilter), ignore_invalid_symbol=true)

# // Oscilator with Volume Symbol Volume Calc
# asset_1 := use_asset_1 ? not na(asset_1) ? asset_1 : request.security( alt_vol_src, res, mode_selector(source_4h_1, oscLength, oscMode, postFilter) ) : na
# asset_2 := use_asset_2 ? not na(asset_2) ? asset_2 : request.security( alt_vol_src, res, mode_selector(source_4h_2, oscLength, oscMode, postFilter) ) : na
# asset_3 := use_asset_3 ? not na(asset_3) ? asset_3 : request.security( alt_vol_src, res, mode_selector(source_4h_3, oscLength, oscMode, postFilter) ) : na
# asset_4 := use_asset_4 ? not na(asset_4) ? asset_4 : request.security( alt_vol_src, res, mode_selector(source_4h_4, oscLength, oscMode, postFilter) ) : na
# asset_5 := use_asset_5 ? not na(asset_5) ? asset_5 : request.security( alt_vol_src, res, mode_selector(source_4h_5, oscLength, oscMode, postFilter) ) : na
# asset_6 := use_asset_6 ? not na(asset_6) ? asset_6 : request.security( alt_vol_src, res, mode_selector(source_4h_6, oscLength, oscMode, postFilter) ) : na
# asset_7 := use_asset_7 ? not na(asset_7) ? asset_7 : request.security( alt_vol_src, res, mode_selector(source_4h_7, oscLength, oscMode, postFilter) ) : na
# asset_8 := use_asset_8 ? not na(asset_8) ? asset_8 : request.security( alt_vol_src, res, mode_selector(source_4h_8, oscLength, oscMode, postFilter) ) : na

# // PLOTS
# zero_plot = plot (0, editable=false, display=display.none)
# plot_1  = plot (asset_1, 'Plot Asset 1',color.new (color_1, 100 - opacity_A), 1)
# fill_1  = plot (asset_1, 'Fill Asset 1',color.new (color_1, 100 - opacity_B), 0, plot.style_area)
# plot_2  = plot (asset_2, 'Plot Asset 2',color.new (color_2, 100 - opacity_A), 1)
# fill_2  = plot (asset_2, 'Fill Asset 2',color.new (color_2, 100 - opacity_B), 0, plot.style_area)
# plot_3  = plot (asset_3, 'Plot Asset 3',color.new (color_3, 100 - opacity_A), 1)
# fill_3  = plot (asset_3, 'Fill Asset 3',color.new (color_3, 100 - opacity_B), 0, plot.style_area)
# plot_4  = plot (asset_4, 'Plot Asset 4',color.new (color_4, 100 - opacity_A), 1)
# fill_4  = plot (asset_4, 'Fill Asset 4',color.new (color_4, 100 - opacity_B), 0, plot.style_area)
# plot_5  = plot (asset_5, 'Plot Asset 5',color.new (color_5, 100 - opacity_A), 1)
# fill_5  = plot (asset_5, 'Fill Asset 5',color.new (color_5, 100 - opacity_B), 0, plot.style_area)
# plot_6  = plot (asset_6, 'Plot Asset 6',color.new (color_6, 100 - opacity_A), 1)
# fill_6  = plot (asset_6, 'Fill Asset 6',color.new (color_6, 100 - opacity_B), 0, plot.style_area)
# plot_7  = plot (asset_7, 'Plot Asset 7',color.new (color_7, 100 - opacity_A), 1)
# fill_7  = plot (asset_7, 'Fill Asset 7',color.new (color_7, 100 - opacity_B), 0, plot.style_area)
# plot_8  = plot (asset_8, 'Plot Asset 8',color.new (color_8, 100 - opacity_A), 1)
# fill_8  = plot (asset_8, 'Fill Asset 8',color.new (color_8, 100 - opacity_B), 0, plot.style_area)

# // LABELS
# var label label_1  = use_asset_1 ? label.new(bar_index+8, asset_1, desc_1, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_1, size=size.small) : na
# var label label_2  = use_asset_2 ? label.new(bar_index+8, asset_2, desc_2, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_2, size=size.small) : na
# var label label_3  = use_asset_3 ? label.new(bar_index+8, asset_3, desc_3, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_3, size=size.small) : na
# var label label_4  = use_asset_4 ? label.new(bar_index+8, asset_4, desc_4, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_4, size=size.small) : na
# var label label_5  = use_asset_5 ? label.new(bar_index+8, asset_5, desc_5, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_5, size=size.small) : na
# var label label_6  = use_asset_6 ? label.new(bar_index+8, asset_6, desc_6, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_6, size=size.small) : na
# var label label_7  = use_asset_7 ? label.new(bar_index+8, asset_7, desc_7, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_7, size=size.small) : na
# var label label_8  = use_asset_8 ? label.new(bar_index+8, asset_8, desc_8, xloc  = xloc.bar_index, style=label.style_label_left, textcolor=color.black,color =color_8, size=size.small) : na

# label.set_xy(label_1, bar_index+4, asset_1)
# label.set_xy(label_2, bar_index+4, asset_2)
# label.set_xy(label_3, bar_index+4, asset_3)
# label.set_xy(label_4, bar_index+4, asset_4)
# label.set_xy(label_5, bar_index+4, asset_5)
# label.set_xy(label_6, bar_index+4, asset_6)
# label.set_xy(label_7, bar_index+4, asset_7)
# label.set_xy(label_8, bar_index+4, asset_8)
