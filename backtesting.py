import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import ta
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
from performance_tracker import PerformanceTracker

# Load environment variables
load_dotenv()

# Initialize Binance client
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_API_SECRET')
client = Client(api_key, api_secret)

def get_historical_data(symbol, interval, start_str, end_str):
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df

def calculate_indicators(df):
    df['SMA_short'] = ta.trend.sma_indicator(df['close'], window=10)
    df['SMA_long'] = ta.trend.sma_indicator(df['close'], window=30)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'])
    bb_indicator = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BB_upper'] = bb_indicator.bollinger_hband()
    df['BB_middle'] = bb_indicator.bollinger_mavg()
    df['BB_lower'] = bb_indicator.bollinger_lband()
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    return df

def generate_signal(df):
    signal = 0
    sma_diff = (df['SMA_short'] - df['SMA_long']) / df['SMA_long']
    signal += sma_diff * 15
    
    if df['RSI'] < 30:
        signal += 2.5
    elif df['RSI'] > 70:
        signal -= 2.5
    
    macd_strength = df['MACD'] / df['close']
    signal += macd_strength * 10
    
    bb_width = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    if df['close'] < df['BB_lower']:
        signal += 1.5
    elif df['close'] > df['BB_upper']:
        signal -= 1.5
    
    return signal

def backtest(symbol, start_date, end_date, interval, initial_balance=10000, risk_percentage=0.01):
    df = get_historical_data(symbol, interval, start_date, end_date)
    df = calculate_indicators(df)
    
    balance = {'USD': initial_balance, 'BTC': 0}
    performance_tracker = PerformanceTracker()
    current_position = {'side': None, 'entry_price': 0, 'quantity': 0}
    
    for i in range(len(df)):
        current_price = df['close'].iloc[i]
        signal = generate_signal(df.iloc[i])
        
        if signal > 0.5 and current_position['side'] != 'long':
            # Buy signal
            quantity = (balance['USD'] * risk_percentage) / current_price
            if balance['USD'] >= quantity * current_price:
                balance['USD'] -= quantity * current_price
                balance['BTC'] += quantity
                current_position = {'side': 'long', 'entry_price': current_price, 'quantity': quantity}
                performance_tracker.log_trade('BUY', current_price, quantity, quantity * current_price)
        
        elif signal < -0.5 and current_position['side'] == 'long':
            # Sell signal
            sell_amount = current_position['quantity'] * current_price
            balance['USD'] += sell_amount
            balance['BTC'] = 0
            profit_loss = (current_price - current_position['entry_price']) / current_position['entry_price']
            performance_tracker.log_trade('SELL', current_price, current_position['quantity'], sell_amount, profit_loss)
            current_position = {'side': None, 'entry_price': 0, 'quantity': 0}
        
        # Update performance tracker
        total_balance = balance['USD'] + balance['BTC'] * current_price
        performance_tracker.calculate_daily_returns(initial_balance, total_balance)
    
    return performance_tracker, balance, df

def plot_results(df, performance_tracker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot price and signals
    ax1.plot(df.index, df['close'], label='Price')
    buy_signals = [trade for trade in performance_tracker.trades if trade[1] == 'BUY']
    sell_signals = [trade for trade in performance_tracker.trades if trade[1] == 'SELL']
    ax1.scatter([trade[0] for trade in buy_signals], [trade[2] for trade in buy_signals], color='green', marker='^', label='Buy')
    ax1.scatter([trade[0] for trade in sell_signals], [trade[2] for trade in sell_signals], color='red', marker='v', label='Sell')
    ax1.set_title('Price and Signals')
    ax1.legend()
    
    # Plot cumulative returns
    cumulative_returns = np.cumprod(1 + np.array([r[1] for r in performance_tracker.daily_returns])) - 1
    
    # Ensure that cumulative_returns has the same length as df.index
    if len(cumulative_returns) < len(df.index):
        # Pad cumulative_returns with NaN values at the beginning
        padding = [np.nan] * (len(df.index) - len(cumulative_returns))
        cumulative_returns = np.concatenate([padding, cumulative_returns])
    elif len(cumulative_returns) > len(df.index):
        # Trim cumulative_returns to match df.index length
        cumulative_returns = cumulative_returns[-len(df.index):]
    
    ax2.plot(df.index, cumulative_returns * 100)
    ax2.set_title('Cumulative Returns (%)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    symbol = "BTCUSDT"
    start_date = "1 Jan, 2023"
    end_date = "1 Jul, 2023"
    interval = Client.KLINE_INTERVAL_1HOUR
    
    performance_tracker, final_balance, df = backtest(symbol, start_date, end_date, interval)
    
    print(f"Final balance: ${final_balance['USD']:.2f} USD, {final_balance['BTC']:.6f} BTC")
    print(f"Total profit/loss: ${performance_tracker.get_total_profit_loss():.2f}")
    
    sharpe_ratio = performance_tracker.calculate_sharpe_ratio()
    print(f"Sharpe ratio: {sharpe_ratio:.4f}" if sharpe_ratio is not None else "Sharpe ratio: Not available")
    
    max_drawdown = performance_tracker.calculate_max_drawdown()
    print(f"Max drawdown: {max_drawdown:.2%}" if max_drawdown is not None else "Max drawdown: Not available")
    
    win_loss_ratio = performance_tracker.calculate_win_loss_ratio()
    print(f"Win/Loss ratio: {win_loss_ratio:.2f}" if win_loss_ratio is not None else "Win/Loss ratio: Not available")
    
    plot_results(df, performance_tracker)

