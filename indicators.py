import numpy as np
import pandas as pd
import ta
import logging

logger = logging.getLogger(__name__)

def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators"""
    try:
        # RSI
        df['RSI'] = ta.momentum.rsi(df['close'], window=14)
        df['RSI_slow'] = ta.momentum.rsi(df['close'], window=21)
        
        # Stochastic RSI
        df['stoch_rsi'] = ta.momentum.stochrsi(df['close'])
        df['stoch_rsi_d'] = ta.momentum.stochrsi_d(df['close'])
        df['stoch_rsi_k'] = ta.momentum.stochrsi_k(df['close'])
        
        # TSI (True Strength Index)
        df['tsi'] = ta.momentum.tsi(df['close'])
        
        # PPO (Percentage Price Oscillator)
        df['ppo'] = ta.momentum.ppo(df['close'])
        df['ppo_signal'] = ta.momentum.ppo_signal(df['close'])
        df['ppo_hist'] = ta.momentum.ppo_hist(df['close'])
        
        return df
    except Exception as e:
        logger.error(f"Error adding momentum indicators: {str(e)}")
        return df

def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators"""
    try:
        # Moving Averages
        df['SMA_short'] = ta.trend.sma_indicator(df['close'], window=10)
        df['SMA_long'] = ta.trend.sma_indicator(df['close'], window=30)
        df['EMA_short'] = ta.trend.ema_indicator(df['close'], window=9)
        df['EMA_long'] = ta.trend.ema_indicator(df['close'], window=21)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # ADX
        df['ADX'] = ta.trend.adx(df['high'], df['low'], df['close'])
        df['ADX_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
        df['ADX_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
        
        # Ichimoku
        df['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
        df['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
        df['ichimoku_base'] = ta.trend.ichimoku_base_line(df['high'], df['low'])
        
        return df
    except Exception as e:
        logger.error(f"Error adding trend indicators: {str(e)}")
        return df

def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators"""
    try:
        # ATR
        df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        df['NATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close']) / df['close']
        
        # Bollinger Bands
        for window in [20, 50]:
            bb = ta.volatility.BollingerBands(df['close'], window=window)
            df[f'BB_upper_{window}'] = bb.bollinger_hband()
            df[f'BB_middle_{window}'] = bb.bollinger_mavg()
            df[f'BB_lower_{window}'] = bb.bollinger_lband()
            df[f'BB_width_{window}'] = (df[f'BB_upper_{window}'] - df[f'BB_lower_{window}']) / df[f'BB_middle_{window}']
        
        # Keltner Channels
        df['keltner_high'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
        df['keltner_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
        
        return df
    except Exception as e:
        logger.error(f"Error adding volatility indicators: {str(e)}")
        return df

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume indicators"""
    try:
        # Basic volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_change'] = df['volume'].pct_change()
        
        # Advanced volume indicators
        df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['CMF'] = ta.volume.chaikin_money_flow(df['high'], df['low'], df['close'], df['volume'])
        df['MFI'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # VWAP
        df['VWAP'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        
        # Volume-based trend strength
        df['volume_trend'] = df['volume'] * (df['close'] - df['close'].shift(1)).apply(np.sign)
        
        return df
    except Exception as e:
        logger.error(f"Error adding volume indicators: {str(e)}")
        return df

def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add support and resistance levels"""
    try:
        df['rolling_high'] = df['high'].rolling(window=window).max()
        df['rolling_low'] = df['low'].rolling(window=window).min()
        
        # Calculate price position relative to support/resistance
        price_range = df['rolling_high'] - df['rolling_low']
        df['price_position'] = (df['close'] - df['rolling_low']) / price_range
        
        return df
    except Exception as e:
        logger.error(f"Error adding support/resistance levels: {str(e)}")
        return df

def calculate_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Calculate all technical indicators"""
    try:
        logger.debug(f"Calculating indicators for {timeframe}")
        
        # Add all indicator groups
        df = add_momentum_indicators(df)
        df = add_trend_indicators(df)
        df = add_volatility_indicators(df)
        df = add_volume_indicators(df)
        df = add_support_resistance(df)
        
        # Add market regime indicators
        df['trend_strength'] = df['ADX']
        df['volatility_regime'] = pd.qcut(df['NATR'].fillna(0), q=3, labels=['low', 'medium', 'high'])
        df['volume_regime'] = pd.qcut(df['volume'].fillna(0), q=3, labels=['low', 'medium', 'high'])
        
        logger.debug(f"Successfully calculated indicators for {timeframe}")
        return df
        
    except Exception as e:
        logger.error(f"Error calculating indicators for {timeframe}: {str(e)}")
        return df

