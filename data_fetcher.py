import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging
import asyncio
import aiohttp
import json
import platform
import numpy as np
import ta
from binance import AsyncClient
from indicators import calculate_indicators


# Create a global Binance client instance
_binance_client = None

async def _create_binance_client():
    global _binance_client
    try:
        _binance_client = await AsyncClient.create()
    except Exception as e:
        logger.error(f"Error creating Binance client: {e}")

async def _close_binance_client():
    global _binance_client
    if _binance_client:
        await _binance_client.close_connection()

logger = logging.getLogger(__name__)

# Windows-specific event loop policy
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def add_indicators(df):
    """Add basic technical indicators"""
    try:
        # Ensure we have enough data
        if len(df) < 20:
            return df
            
        # Basic indicators
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['RSI'] = ta.momentum.rsi(df['close'])
        df['MACD'] = ta.trend.macd_diff(df['close'])
        
        # Fill any NaN values using newer methods
        df = df.bfill().fillna(0)
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return df

async def get_historical_data_async(symbol, interval, start_str=None, limit=1000):
    """Async version of historical data fetching"""
    try:
        # Request more data than needed to ensure enough for indicators
        actual_limit = max(limit + 100, 1000)
        endpoint = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={actual_limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                klines = await response.json()
                
                if not klines:
                    return pd.DataFrame()
                
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                
                # Convert to proper types
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Add indicators
                df = add_indicators(df)
                
                # Return only the requested amount of data
                return df.tail(limit)
                
    except Exception as e:
        logger.error(f"Async data fetch error: {e}")
        return pd.DataFrame()

async def get_current_price_async(symbol):
    """Async version of current price fetching"""
    try:
        endpoint = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                data = await response.json()
                return float(data['price'])
    except Exception as e:
        logger.error(f"Async price fetch error: {e}")
        return None

async def get_current_price(client, symbol: str) -> float:
    """Get current price for a symbol"""
    
    global _binance_client
    try:
        ticker = await _binance_client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except BinanceAPIException as e:
        logger.error(f"Binance API error fetching current price for {symbol}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching current price for {symbol}: {e}")

    # Fallback to aiohttp if Binance client fails
    return await get_current_price_async(symbol)

async def get_historical_data(client, symbol: str, interval: str) -> pd.DataFrame:
    """Get historical klines/candlestick data"""
    try:
        # Use the async data fetcher for better performance and error handling
        df = await get_historical_data_async(symbol, interval, limit=1000)
        if df.empty:
            logger.error(f"Failed to fetch historical data for {symbol} {interval}")
            return df  # Return the empty DataFrame
        
        # async_client = await AsyncClient.create()
        # # Get the timestamp for start time (e.g., last 1000 candles)
        # klines = await async_client.get_klines(
        #     symbol=symbol,
        #     interval=interval,
        #     limit=1000
        # )
        # await async_client.close_connection()
        
        # Create DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Convert strings to floats
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        # Calculate indicators
        # df = calculate_indicators(df, interval) # Already calculated in get_historical_data_async
        
        return df
        
    except (BinanceAPIException, aiohttp.ClientError) as e:
        logger.error(f"Error fetching historical data for {symbol} {interval}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Create the Binance client when the module is imported
asyncio.run(_create_binance_client())

# Close the client when the script exits
import atexit; atexit.register(lambda: asyncio.run(_close_binance_client()))
