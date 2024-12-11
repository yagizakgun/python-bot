import logging
import asyncio
import numpy as np
from data_fetcher import get_historical_data, get_current_price
from datetime import datetime, timedelta
from data_fetcher import get_historical_data_async, get_current_price_async

logger = logging.getLogger(__name__)

def calculate_current_volatility(df):
    """Calculate current market volatility using ATR-based method"""
    try:
        # Use ATR-based volatility if available
        if 'ATR' in df.columns:
            return df['ATR'].iloc[-1] / df['close'].iloc[-1]
        # Fallback to standard deviation
        return df['close'].pct_change().std()
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return 0.01

def analyze_trend_strength(df):
    """Analyze trend strength using multiple indicators"""
    try:
        # Use EMA crossovers
        df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Trend direction
        trend_direction = 1 if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else -1
        
        # Trend strength based on distance between EMAs
        trend_strength = abs(df['EMA20'].iloc[-1] - df['EMA50'].iloc[-1]) / df['EMA50'].iloc[-1]
        
        # Volume trend
        volume_trend = df['volume'].ewm(span=20, adjust=False).mean().iloc[-1] > df['volume'].ewm(span=50, adjust=False).mean().iloc[-1]
        
        # RSI for confirmation
        rsi_trend = 1 if df['RSI'].iloc[-1] > 50 else -1
        
        # Combine factors
        strength = trend_strength * trend_direction * (1.2 if volume_trend else 0.8) * rsi_trend
        return strength
        
    except Exception as e:
        logger.error(f"Error analyzing trend: {e}")
        return 0

def calculate_signal_strength(df):
    """Calculate trading signal strength using multiple factors"""
    try:
        # Price momentum
        momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        # RSI divergence
        rsi = df['RSI'].iloc[-1]
        rsi_signal = (rsi - 50) / 50  # Normalize RSI to -1 to 1
        
        # MACD signal
        macd_signal = df['MACD'].iloc[-1] / df['close'].iloc[-1]  # Normalize by price
        
        # Volume confirmation
        volume_factor = 1.2 if df['volume'].iloc[-1] > df['volume'].rolling(20).mean().iloc[-1] else 0.8
        
        # ATR for volatility adjustment
        atr_factor = 1.0
        if 'ATR' in df.columns:
            atr = df['ATR'].iloc[-1] / df['close'].iloc[-1]
            atr_factor = 1.2 if atr < 0.01 else (0.8 if atr > 0.03 else 1.0)
        
        # Combine signals with weights
        signal = (
            momentum * 0.3 +
            rsi_signal * 0.3 +
            macd_signal * 0.4
        ) * volume_factor * atr_factor
        
        return np.clip(signal, -1, 1)
        
    except Exception as e:
        logger.error(f"Error calculating signal strength: {e}")
        return 0

async def analyze_multiple_timeframes_async(symbol, timeframes, config):
    """Async version of timeframe analysis with enhanced signal generation"""
    try:
        tasks = [get_historical_data_async(symbol, tf) for tf in timeframes]
        dataframes = await asyncio.gather(*tasks)
        
        signals = []
        weights = []  # Different weights for different timeframes
        
        for df, tf in zip(dataframes, timeframes):
            if df.empty:
                continue
                
            # Calculate base signal
            signal = calculate_signal_strength(df)
            
            # Adjust weight based on timeframe
            if tf in ['1m', '3m']:
                weight = 0.1  # Lower weight for very short timeframes
            elif tf in ['5m', '15m']:
                weight = 0.2
            elif tf in ['30m', '1h']:
                weight = 0.3
            else:
                weight = 0.4  # Higher weight for longer timeframes
            
            # Adjust signal based on trend
            trend_strength = analyze_trend_strength(df)
            signal *= (1 + abs(trend_strength))
            
            signals.append(signal)
            weights.append(weight)
        
        if not signals:
            return [], "undefined"
            
        # Calculate weighted average signal
        weights = np.array(weights) / sum(weights)  # Normalize weights
        combined = np.average(signals, weights=weights)
        
        # Determine market condition
        volatility = calculate_current_volatility(dataframes[0])  # Use shortest timeframe for volatility
        market_condition = "volatile_trending" if abs(combined) > 0.3 and volatility > 0.02 else \
                         "trending" if abs(combined) > 0.3 else \
                         "volatile_ranging" if volatility > 0.02 else "ranging"
        
        return signals, market_condition
        
    except Exception as e:
        logger.error(f"Error in timeframe analysis: {e}")
        return [], "undefined"

def combine_signals(signals, volatility, market_condition):
    """Combine signals with advanced filtering"""
    try:
        # Handle both dictionary and tuple/list formats
        if isinstance(signals, dict):
            signal_values = list(signals.values())
        elif isinstance(signals, (tuple, list)):
            signal_values = list(signals)
        else:
            logger.error(f"Unexpected signals type: {type(signals)}")
            return 0, 0.5, -0.5
            
        if not signal_values:
            return 0, 0.5, -0.5
            
        # Calculate weighted signal
        combined = sum(signal_values) / len(signal_values)
        
        # Dynamic thresholds based on market conditions
        base_threshold = 0.5 * (1 + volatility)
        
        # Adjust thresholds based on market condition
        if market_condition == "volatile_trending":
            base_threshold *= 1.2  # Higher threshold in volatile trending markets
        elif market_condition == "ranging":
            base_threshold *= 0.8  # Lower threshold in ranging markets
        
        # Asymmetric thresholds
        buy_threshold = base_threshold
        sell_threshold = -base_threshold * 1.1  # Slightly higher threshold for sells
        
        return combined, buy_threshold, sell_threshold
        
    except Exception as e:
        logger.error(f"Error combining signals: {e}")
        return 0, 0.5, -0.5

async def get_market_analysis_async(symbol, interval='1h'):
    """Async version of market analysis with enhanced metrics"""
    try:
        df = await get_historical_data_async(symbol, interval)
        if df.empty:
            return {'condition': 'undefined', 'structure': {}}
        
        volatility = calculate_current_volatility(df)
        trend_strength = analyze_trend_strength(df)
        
        # Determine trend direction with confirmation
        price_trend = 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-10] else 'bearish'
        ema_trend = 'bullish' if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else 'bearish'
        
        # Only confirm trend if both indicators agree
        trend = price_trend if price_trend == ema_trend else 'mixed'
        
        # Classify volatility
        vol_description = (
            'high' if volatility > 0.02 else
            'medium' if volatility > 0.01 else
            'low'
        )
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_description = (
            'high' if current_volume > avg_volume * 1.5 else
            'medium' if current_volume > avg_volume * 0.5 else
            'low'
        )
        
        return {
            'condition': f"{trend}_{'volatile' if volatility > 0.02 else 'stable'}",
            'structure': {
                'trend': {
                    'direction': trend,
                    'strength': abs(trend_strength)
                },
                'volatility': {
                    'value': volatility,
                    'description': vol_description,
                    'normalized': f"{volatility:.4f}"
                },
                'volume': {
                    'value': current_volume,
                    'description': volume_description,
                    'relative_to_avg': f"{(current_volume/avg_volume):.2f}x"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        return {'condition': 'undefined', 'structure': {}}

async def analyze_multiple_timeframes(client, symbol, timeframes, config):
    """Analyze multiple timeframes and return signals"""
    try:
        signals = []
        weights = []  # Different weights for different timeframes
        
        # Get data for each timeframe
        for tf in timeframes:
            df = await get_historical_data(client, symbol, tf)
            if df is None or df.empty:
                continue
                
            # Calculate base signal
            signal = calculate_signal_strength(df)
            
            # Adjust weight based on timeframe
            if tf in ['1m', '3m']:
                weight = 0.1  # Lower weight for very short timeframes
            elif tf in ['5m', '15m']:
                weight = 0.2
            elif tf in ['30m', '1h']:
                weight = 0.3
            else:
                weight = 0.4  # Higher weight for longer timeframes
            
            # Adjust signal based on trend
            trend_strength = analyze_trend_strength(df)
            signal *= (1 + abs(trend_strength))
            
            signals.append(signal)
            weights.append(weight)
        
        if not signals:
            return [], "undefined"
            
        # Calculate weighted average signal
        weights = np.array(weights) / sum(weights)  # Normalize weights
        combined = np.average(signals, weights=weights)
        
        # Determine market condition
        volatility = calculate_current_volatility(df)  # Use last timeframe's data for volatility
        market_condition = "volatile_trending" if abs(combined) > 0.3 and volatility > 0.02 else \
                         "trending" if abs(combined) > 0.3 else \
                         "volatile_ranging" if volatility > 0.02 else "ranging"
        
        return signals, market_condition
        
    except Exception as e:
        logger.error(f"Error in timeframe analysis: {e}")
        return [], "undefined"

def get_market_analysis(df, config):
    """Analyze market data"""
    try:
        if df is None or df.empty:
            return {'condition': 'undefined', 'structure': {}}
        
        volatility = calculate_current_volatility(df)
        trend_strength = analyze_trend_strength(df)
        
        # Determine trend direction with confirmation
        price_trend = 'bullish' if df['close'].iloc[-1] > df['close'].iloc[-10] else 'bearish'
        ema_trend = 'bullish' if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else 'bearish'
        
        # Only confirm trend if both indicators agree
        trend = price_trend if price_trend == ema_trend else 'mixed'
        
        # Classify volatility
        vol_description = (
            'high' if volatility > 0.02 else
            'medium' if volatility > 0.01 else
            'low'
        )
        
        # Volume analysis
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        current_volume = df['volume'].iloc[-1]
        volume_description = (
            'high' if current_volume > avg_volume * 1.5 else
            'medium' if current_volume > avg_volume * 0.5 else
            'low'
        )
        
        return {
            'condition': f"{trend}_{'volatile' if volatility > 0.02 else 'stable'}",
            'structure': {
                'trend': {
                    'direction': trend,
                    'strength': abs(trend_strength)
                },
                'volatility': {
                    'value': volatility,
                    'description': vol_description,
                    'normalized': f"{volatility:.4f}"
                },
                'volume': {
                    'value': current_volume,
                    'description': volume_description,
                    'relative_to_avg': f"{(current_volume/avg_volume):.2f}x"
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Error in market analysis: {e}")
        return {'condition': 'undefined', 'structure': {}}

