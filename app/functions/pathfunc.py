import os

def get_path_app() :
    return "app"

def get_path_data() :
    return "data"

def get_path_crypto() :
    return os.path.join( get_path_data(), "crypto_binance")

def get_path_sp500() :
    return os.path.join( get_path_data(), "sp500")
