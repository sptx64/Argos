# Argos
## a trading analysis stock explorer

Argos is a Streamlit-based application that empowers you to download, screen, and analyze financial data. It supports:
- S&P 500 and Euronext stock data via the yfinance API.
- US-authorized cryptocurrencies through the Binance API.

Built purely out of a passion for coding, this app is for educational purposes only—use it at your own risk, as it’s not intended for financial advice.

***

Key Features

1. Hermes: The Data Messenger
The Hermes module manages database updates by fetching data from yfinance or Binance APIs.
Note: Data retrieval can be time-consuming and may hit yfinance API rate limits. Adjust the time.sleep(x) parameter to avoid issues.


2. Argos: The Core Screener
The Argos module is the heart of the app, allowing you to explore the full dataset of stocks and cryptocurrencies by bcreen data using technical indicators and custom setups,such as RSI (Relative Strength Index), Bollinger Bands (BB), SMA/EMA (Simple/Exponential Moving Averages), Doji Patterns and custom screening functions.

Disclaimer: Built-in functions are provided as-is, and results are not guaranteed. Always verify your analysis.


3. Zeus: The Stock Focus Analysis
Once you’ve identified a stock or crypto of interest, the Zeus page lets you zoom in for detailed visualizations and insights.

***

Deployment Options - two ways to use Argos

Fork and Deploy on Streamlit Community Cloud:
Fork the repository and deploy your own instance. You do not need any coding experience.
Note: Limited cryptocurrency data and potential yfinance rate limits apply.

Run Locally:
Enjoy access to more cryptocurrencies and bypass yfinance rate limits.
Follow the setup instructions below to get started.

Clone the repository:
git clone https://github.com/sptx64/argos.git

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app locally:
streamlit run app.py

***

Passwords are in the .streamlit/secrets.toml file

***

This project is a passion project, built for the love of coding.
Use at your own risk. It is not financial advice—always conduct your own research.

***

Contributing
Contributions are welcome! To contribute:
Report bugs or suggest features via Issues.
Submit pull requests with enhancements or fixes.

***

License
This project is licensed under the MIT License. See the LICENSE file for details.
Built with ❤️
Explore, analyze, and experiment responsibly!