import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates
from datetime import datetime
import uuid
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os  # Add this import to access environment variables

st.set_page_config(page_title="Stock Simulator", layout="wide")
# testing diary
st.write("✅ Test code loaded")
print("✅ Test code loaded")

# code testing
st.title("💹 Stock Simulator")
st.write("This is a stock simulator that mimics people's behavior in stocks.")

# ...existing code...
try:
    st.write("🛠️ operations begin running…")
except Exception as e:
    st.error(f"🚨 failed：{e}")

from typing import List
import pandas as pd

def get_stock_data(ticker: str, start_date: str, end_date: str, columns: List[str] = ['Close']) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data[columns]
    except Exception as e:
        st.error(f"🚨 Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame to avoid errors in the subsequent code.

valid_tickers = ['AAPL', 'TSLA', 'MSFT']
stock_data_dict = {}
ticker = None 
for ticker in valid_tickers:
    df = get_stock_data(ticker, "2023-01-03", "2023-3-30", ["Close"])
    df = df.reset_index()
    stock_data_dict[ticker] = df

class User:
    def __init__(self, cash: float):
        if cash <= 0:
            st.warning("⚠️ Not enough cash.")
            return
        self.cash = cash
        self.stocks = {}
        self.history = []
        self.profit = 0.0
    def __repr__(self):
        return f"User(cash=${self.cash}, stocks={self.stocks})"

import uuid

def buy_stock(user:User, ticker, price, shares):
    # Input validation: price and shares must be positive
    if shares <= 0 or price <= 0:
        raise ValueError("❌ Price and number of shares must be greater than 0.")

    # Calculate cost and trading fee (1%)
    cost = price * shares
    fee = cost * 0.01
    total_cost = cost + fee

    if user.cash >= total_cost:
        user.cash -= total_cost
        # ✅ If first time buying this stock
        if ticker not in user.stocks:
            user.stocks[ticker] = {
                'shares': shares,
                'avg_price': price
            }
        else:
            # ✅ Update average price
            prev_data = user.stocks[ticker]
            prev_shares = prev_data['shares']
            prev_avg = prev_data['avg_price']
            total_shares = prev_shares + shares
            new_avg = (prev_avg * prev_shares + price * shares) / total_shares

            user.stocks[ticker]['shares'] = total_shares
            user.stocks[ticker]['avg_price'] = round(new_avg, 2)

        # Record transaction, including unique ID
        user.history.append({
            'id': str(uuid.uuid4()),
            'type': 'buy',
            'ticker': ticker,
            'price': price,
            'shares': shares,
            'fee': round(fee, 2),
            'total': round(total_cost, 2),
            'time': datetime.now()
        })

        st.success(f"✅ Bought {shares} shares of {ticker} at ${price:.2f}")

        # Low balance warning
        if user.cash < 100:
            st.warning(f"⚠️ Warning: Cash balance is low (${user.cash:.2f}).")
            return

    else:
        st.warning("❌ Purchase failed: insufficient funds.")

# Check if the user owns the stock and has enough shares
def sell_stock(user, ticker, price, shares):
    if ticker not in user.stocks:
        st.warning(f"❌ You do not own any shares of {ticker}.")
        return
    if user.stocks[ticker]['shares'] < shares:
        st.warning(f"❌ Not enough shares to sell. You own {user.stocks[ticker]['shares']} shares.")
        return
    if ticker in user.stocks and user.stocks[ticker]['shares'] >= shares:

        # Retrieve original cost information
        stock_info = user.stocks[ticker]
        avg_price = stock_info['avg_price']
        current_shares = stock_info['shares']

    # Update holdings
        stock_info['shares'] -= shares
        revenue = price * shares
        profit = (price - avg_price) * shares
        user.cash += revenue
        user.profit += profit

    # Auto-remove the stock entry if no shares left
    if stock_info['shares'] == 0:
        del user.stocks[ticker]

    # Add transaction record
    user.history.append({
        'id': str(uuid.uuid4()),
        'type': 'sell',
        'ticker': ticker,
        'price': price,
        'shares': shares,
        'profit': round(profit, 2),
        'time': datetime.now()
    })

def plot_portfolio_value_interactive(history_df):
    # change Date Style
    history_df['time'] = pd.to_datetime(history_df['time'])

    # Create Plotly Graph
    fig = px.line(
        history_df,
        x='time',
        y='total_value',
        title="📈 Portfolio Total Value Over Time",
        labels={'time': 'Date', 'total_value': 'Value ($)'},
        markers=True
    )

    # Make the Graph Better
    fig.update_layout(
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        hovermode='x unified',
        template='plotly_dark',
        autosize=True
    )

    # Show figure
    fig.show()



def export_transaction_history(user, filename=None, start_date=None, end_date=None, transaction_type=None, ticker=None):
    """
    Export the user's transaction history to a CSV or Excel file.

    Parameters:
    - user: dict, should contain 'history' as a list of transactions.
    - filename: str, optional. If not provided, a timestamped name will be generated.
    - start_date: str, optional. Format 'YYYY-MM-DD'. Filters transactions from this date onward.
    - end_date: str, optional. Format 'YYYY-MM-DD'. Filters transactions up to this date.
    - transaction_type: str, optional. 'buy' or 'sell' to filter by type.
    - ticker: str, optional. Filter by stock ticker symbol.
    """

    df = pd.DataFrame(user['history'])
    if df.empty:
        print("⚠️ No transaction history available to export.")
        return

    # Ensure time column is datetime
    df['time'] = pd.to_datetime(df['time'])

    # Filter by date range
    if start_date:
        df = df[df['time'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['time'] <= pd.to_datetime(end_date)]

    # Filter by transaction type
    if transaction_type:
        df = df[df['type'] == transaction_type]

    # Filter by ticker
    if ticker:
        df = df[df['ticker'] == ticker]

    # Generate default filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transaction_history_{timestamp}.csv"

    # Export to CSV or Excel based on filename extension
    if filename.endswith('.xlsx'):
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)

    print(f"✅ Transaction history has been saved to: {filename}")

from tabulate import tabulate

def print_portfolio_summary(user, current_price_dict):
    """
    Print a detailed portfolio summary for the user.

    Parameters:
    - user: dict with keys 'cash', 'stocks', 'history', etc.
    - current_price_dict: dict of latest stock prices (e.g., {'AAPL': 172.8})
    """
    print("\n📊 Portfolio Summary:")
    print(f"Available Cash: ${user['cash']:.2f}")

    if not user['stocks']:
        print("No stocks currently held.")
        return

    table = []
    total_value = user['cash']

    for ticker, info in user['stocks'].items():
        shares = info['shares']
        avg_price = info['avg_price']
        current_price = current_price_dict.get(ticker, 0.0)
        market_value = shares * current_price
        unrealized_profit = (current_price - avg_price) * shares

        total_value += market_value

        table.append([
            ticker,
            shares,
            f"${avg_price:.2f}",
            f"${current_price:.2f}",
            f"${market_value:.2f}",
            f"${unrealized_profit:.2f}"
        ])

    print("\n📈 Stock Holdings:")
    print(tabulate(table, headers=["Ticker", "Shares", "Avg Buy Price", "Current Price", "Market Value", "P/L"], tablefmt="pretty"))
    print(f"\n💼 Total Account Value: ${total_value:.2f}")

st.title("✅ Streamlit Successfully Begins！")
st.write("Welcome to your first Stock Simulation Page 🎈")

# Initialize user as an instance of the User class
cash_input = st.number_input("💰 Enter initial cash:", min_value=0.0, value=10000.0, step=100.0)

# Update or initialize session state for the user
if "user" not in st.session_state or st.session_state.user.cash != cash_input:
    st.session_state.user = User(cash=cash_input)

user = st.session_state.user  # Use the persisted user object

# Debugging: Log the initial state of user
st.write("Initialized User:", user)

# Date range
start = '2023-01-01'
end = '2023-03-31'
ticker = st.selectbox("📈 Choose a stock:", options=valid_tickers)

# Fetch historical data
if ticker:
    with st.spinner("📦 Fetching historical stock data..."):
        data = get_stock_data(ticker, start, end)
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date'])
        st.success(f"✅ Loaded {ticker} data from {start} to {end}")
        st.line_chart(data.set_index('Date')['Close'])

from newsapi import NewsApiClient

# Replace 'YOUR_NEWSAPI_KEY' with your actual NewsAPI key
NEWS_API_KEY = os.getenv('NEWS_API_KEY')  # Fetch key from environment variables

# Initialize NewsAPI client with fallback mechanism
try:
    if not NEWS_API_KEY:
        st.warning("⚠️ NEWS_API_KEY is not set. Please enter it below.")
        NEWS_API_KEY = st.text_input("🔑 Enter your NewsAPI Key:", type="password")
        if not NEWS_API_KEY:
            raise ValueError("NEWS_API_KEY is still not provided. Cannot proceed.")
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    st.success("✅ NewsAPI client initialized successfully.")
except Exception as e:
    st.error(f"🚨 Failed to initialize NewsAPI client: {e}")
    newsapi = None

if ticker and newsapi:
    st.markdown("### 📰 Latest News")
    try:
        # Fetch news articles related to the stock ticker
        articles = newsapi.get_everything(q=ticker, language='en', sort_by='publishedAt', page_size=5)
        if articles and articles.get('articles'):
            for article in articles['articles']:
                title = article.get('title', 'No Title Available')
                link = article.get('url', '#')  # Get the news link
                st.markdown(f"- [{title}]({link})" if link and link != '#' else f"- {title} (No link available)")
        else:
            st.info("ℹ️ No news available for this stock.")
    except Exception as e:
        st.error(f"🚨 Failed to fetch news for {ticker}: {e}")
        st.write("Debug Info:", e)

# Debugging: Log the fetched data
st.dataframe(data)

# Initialize session state for simulation
if "day" not in st.session_state:
    st.session_state.day = 0

# Simulation logic
i = st.session_state.day
if i >= len(data):
    st.success("✅ Simulation complete!")
    history_df = pd.DataFrame(user.history)
    if not history_df.empty:
        final_value = history_df['total_value'].iloc[-1]
        return_pct = (final_value - user.cash) / user.cash * 100
        st.write(f"📈 Total return: {return_pct:.2f}%")
    else:
        st.warning("⚠️ No transaction history available.")
else:
    row = data.iloc[i]

    # Ensure row['Date'] and row['Close'] are scalar values
    date = row['Date'] if not isinstance(row['Date'], pd.Series) else row['Date'].iloc[0]
    price = row['Close'] if not isinstance(row['Close'], pd.Series) else row['Close'].iloc[0]

    st.markdown(f"### 📅 Date: {date.strftime('%Y-%m-%d')} | 💰 Current Price: ${price:.2f}")

    # Provide trading form
    with st.form("trading_form"):
        action = st.selectbox("Choose your action", ['buy', 'sell', 'hold', 'quit'])
        shares = st.number_input("Number of shares", min_value=0, step=1)
        submitted = st.form_submit_button("Submit action")

    if submitted:
        if action == 'buy':
            buy_stock(user, ticker, price, shares)
        elif action == 'sell':
            sell_stock(user, ticker, price, shares)
        elif action == 'hold':
            st.info("🤖 Holding position. No action taken.")
        elif action == 'quit':
            st.warning("❌ Exiting simulation.")
        else:
            st.warning("⚠️ Invalid action. Skipping this day.")

        # Update history records
        total_value = user.cash + sum(
            info['shares'] * price for ticker, info in user.stocks.items()
        )
        user.history.append({
            'time': date,
            'total_value': total_value
        })

        # Persist the updated user object
        st.session_state.user = user

        # Debugging: Log the updated user and history
        st.write("Updated User:", user)
        st.write("History Records:", user.history)
        # Calculate total assets and profit
        total_assets = user.cash + sum(
            info['shares'] * price for ticker, info in user.stocks.items()
        )
        profit = total_assets - cash_input  # initial cash as cash_input

        # Display current cash and profit/loss
        st.write(f"💰 Current Cash: ${user.cash:.2f}")
        st.write(f"📈 Total Assets: ${total_assets:.2f}")
        st.write(f"📊 Profit/Loss: ${profit:.2f}")

        # Provide trading history download functionality
        if user.history:
            # Convert trading history to a DataFrame
            history_df = pd.DataFrame(user.history)

            # Convert the DataFrame to CSV format
            csv = history_df.to_csv(index=False)

            # Provide a download button
        st.download_button(
            label="📥 Download Transaction History",
            data=csv,
            file_name="transaction_history.csv",
            mime="text/csv",
        )
        if not user.history:
            st.info("No transaction history available to download.")
    if st.button("Exit Market"):
        st.success("🚪 You have exited the market. Thank you for participating!")
        st.stop()  # Stop Streamlit
    if st.button("Next Day"):
        st.session_state.day += 1
    # ...existing code...

# Provide trading history download functionality
if user.history:
    # Convert trading history to a DataFrame
    history_df = pd.DataFrame(user.history)

    # Convert the DataFrame to CSV format
    csv = history_df.to_csv(index=False)

    # Provide a download button
    st.download_button(
        label="📥 Download Transaction History",
        data=csv,
        file_name="transaction_history.csv",
        mime="text/csv",
        key="download_transaction_history"  # Add a unique key
    )
else:
    st.info("No transaction history available to download.")

# Type: streamlit run "<path_to_file>\simplified_stock_trading_system.py"
# in the terminal to display Streamlit.
