import logging
from datetime import datetime, timedelta
import uuid
import json
import csv
from colorama import Fore, Style, init
import numpy as np
from functools import lru_cache
from typing import Dict, Tuple
logger = logging.getLogger(__name__)

def generate_request_id():
    return f"{datetime.now().strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:8]}"

def log_to_json(data, filename="trading_log.json"):
    with open(filename, "a") as f:
        json.dump(data, f)
        f.write("\n")  # Her giriş için yeni bir satır

def log_to_csv(data, filename="trading_log.csv"):
    fieldnames = [
        "timestamp", "request_id", "current_price", "combined_signal",
        "market_condition", "market_structure_trend", "market_structure_volatility",
        "market_structure_volume", "volatility", "atr", "balance_usd", "balance_btc",
        "total_balance_usd", "position_side", "position_quantity", "position_average_price",
        "position_profit_loss", "stop_loss", "take_profit"
    ]
    
    try:
        # Format market structure data
        if isinstance(data.get('market_structure'), dict):
            market_struct = data['market_structure']
            trend_data = market_struct.get('trend', {})
            vol_data = market_struct.get('volatility', {})
            volume_data = market_struct.get('volume', {})
            
            # Safely format trend data
            data['market_structure_trend'] = (
                f"direction={trend_data.get('direction', 'unknown')},"
                f"strength={trend_data.get('strength', 0)}"
            )
            
            # Safely format volatility data
            data['market_structure_volatility'] = (
                f"value={float(vol_data.get('value', 0)):.4f}"
            )
            
            # Safely format volume data
            data['market_structure_volume'] = (
                f"value={float(volume_data.get('value', 0)):.2f}"
            )
        
        # Format position data
        if 'position' in data and isinstance(data['position'], dict):
            position = data['position']
            data['position_side'] = position.get('side', '')
            data['position_quantity'] = position.get('total_quantity', 0)
            data['position_average_price'] = position.get('average_price', 0)
            data['stop_loss'] = position.get('stop_loss', '')
            data['take_profit'] = position.get('take_profit', '')
            
            # Calculate position profit/loss if possible
            if position.get('average_price') and data.get('current_price') and position.get('total_quantity', 0) > 0:
                profit_loss = ((data['current_price'] - position['average_price']) / position['average_price']) * 100
                data['position_profit_loss'] = f"{profit_loss:.2f}%"
            else:
                data['position_profit_loss'] = ''
        
        # Ensure numeric values are properly formatted
        data['current_price'] = f"{float(data.get('current_price', 0)):.2f}"
        data['combined_signal'] = f"{float(data.get('combined_signal', 0)):.4f}"
        data['volatility'] = f"{float(data.get('volatility', 0)):.6f}"
        data['atr'] = f"{float(data.get('atr', 0)):.2f}"
        data['balance_usd'] = f"{float(data.get('balance_usd', 0)):.2f}"
        data['balance_btc'] = f"{float(data.get('balance_btc', 0)):.8f}"
        
        # Calculate total balance in USD with proper precision
        if 'balance_usd' in data and 'balance_btc' in data and 'current_price' in data:
            total_balance = (float(data['balance_usd']) + 
                           float(data['balance_btc']) * float(data['current_price']))
            data['total_balance_usd'] = f"{total_balance:.2f}"
        
        with open(filename, "a", newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if csv_file.tell() == 0:
                writer.writeheader()
            
            # Create a new row with only the defined fieldnames
            row = {field: data.get(field, '') for field in fieldnames}
            writer.writerow(row)
            csv_file.flush()
            
    except Exception as e:
        logger.error(f"CSV writing error: {str(e)}", exc_info=True)
        # Log the error to a separate file for debugging
        log_error(data.get('request_id', 'unknown'), 'CSV_WRITE_ERROR', str(e), str(data))

def print_boxed(text, color):
    box_width = len(text) + 4
    box_top = "+" + "-" * box_width + "+"
    box_middle = f"| {text} |"
    
    logger.info(color + box_top + Style.RESET_ALL)
    logger.info(color + box_middle + Style.RESET_ALL)
    logger.info(color + box_top + Style.RESET_ALL)

@lru_cache(maxsize=128)
def get_condition_multiplier(market_condition: str) -> float:
    """Cached market condition multipliers"""
    multipliers = {
        "volatile_trending": 0.5,
        "volatile_ranging": 0.75,
        "trending": 0.8,
        "ranging": 1.25,
        "undefined": 1.0
    }
    return multipliers.get(market_condition, 1.0)

def calculate_dynamic_cooling_period(
    market_condition: str,
    volatility: float,
    base_period: int,
    signal_strength: float | None = None,
    last_trade_profit: float | None = None
) -> int:
    """Optimized dynamic cooling period calculation"""
    
    # Get cached multiplier
    base_multiplier = get_condition_multiplier(market_condition)
    
    # Vectorized calculations using numpy
    factors = np.array([
        1.5 if volatility < 0.005 else 0.8 if volatility > 0.02 else 1.0,  # volatility
        0.8 if signal_strength and abs(signal_strength) > 0.8 else 1.2 if signal_strength and abs(signal_strength) < 0.3 else 1.0,  # signal
        1.5 if last_trade_profit and last_trade_profit < -1.0 else 0.9 if last_trade_profit and last_trade_profit > 2.0 else 1.0  # performance
    ])
    
    # Calculate final period
    dynamic_period = int(base_period * base_multiplier * np.prod(factors))
    return max(5, dynamic_period)

def can_trade(last_trade_time, market_condition, volatility, cooling_period, 
              signal_strength=None, last_trade_profit=None):
    """
    Determine if trading is allowed based on dynamic cooling period and market conditions.
    
    Args:
        last_trade_time (datetime): Time of the last trade
        market_condition (str): Current market condition
        volatility (float): Market volatility measure
        cooling_period (int): Base cooling period in minutes
        signal_strength (float, optional): Strength of the current signal
        last_trade_profit (float, optional): Profit/loss percentage of the last trade
    
    Returns:
        tuple: (bool, str) - (can trade flag, reason message)
    """
    if last_trade_time is None:
        return True, "First trade - no cooling period applied"
    
    dynamic_period = calculate_dynamic_cooling_period(
        market_condition, volatility, cooling_period,
        signal_strength, last_trade_profit
    )
    
    time_since_last_trade = datetime.now() - last_trade_time
    minutes_since_last_trade = time_since_last_trade.total_seconds() / 60
    
    # Check if we're still in cooling period
    if minutes_since_last_trade < dynamic_period:
        remaining_minutes = dynamic_period - minutes_since_last_trade
        reason = (f"Cooling period active: {remaining_minutes:.1f} minutes remaining. "
                 f"Dynamic period: {dynamic_period} minutes")
        return False, reason
    
    return True, f"Cooling period complete. Time since last trade: {minutes_since_last_trade:.1f} minutes"

def log_queued_signal(signal_data, filename="queued_signals.json"):
    """Log signals that occur during cooling period."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "signal_strength": signal_data.get("strength"),
        "market_condition": signal_data.get("market_condition"),
        "volatility": signal_data.get("volatility"),
        "price": signal_data.get("price"),
        "reason": "Signal during cooling period"
    }
    
    with open(filename, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

def calculate_dynamic_levels(current_price, atr, volatility):
    base_risk_factor = 1.5
    if volatility < 0.005:
        risk_factor = base_risk_factor * 0.75
    elif volatility < 0.01:
        risk_factor = base_risk_factor
    else:
        risk_factor = base_risk_factor * 1.25
    
    stop_loss = current_price - atr * risk_factor
    take_profit = current_price + atr * risk_factor * 1.5
    return stop_loss, take_profit

def log_error(request_id, error_type, error_message, traceback_info, filename="error_log.json"):
    error_data = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "error_type": error_type,
        "error_message": error_message,
        "traceback": traceback_info
    }
    with open(filename, "a") as f:
        json.dump(error_data, f)
        f.write("\n")  # Her hata için yeni bir satır

