import logging
from datetime import datetime, timedelta
from exceptions import TradeExecutionError, InsufficientFundsError
from dataclasses import dataclass
from typing import Dict, List, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    success: bool
    balance: Dict[str, float]
    position: Dict[str, any]
    message: str

def execute_trade(
    action: str,
    price: float,
    atr: float,
    combined_signal: float,
    reasons: List[str],
    stop_loss: float,
    take_profit: float,
    market_condition: str,
    volatility: float,
    request_id: str,
    balance: Dict[str, float],
    current_position: Dict[str, any],
    risk_percentage: float,
    performance_tracker: any
) -> TradeResult:
    """Enhanced trade execution with dynamic position sizing and risk management"""
    try:
        # Log trade reasons
        logger.info(f"[{request_id}] Trade reasons for {action}:")
        for reason in reasons:
            logger.info(f"- {reason}")

        # Early validation with detailed checks
        if not validate_trade_conditions(price, atr, market_condition, volatility, balance):
            return TradeResult(False, balance, current_position, "Trade validation failed")

        # Dynamic position sizing based on multiple factors
        position_size = calculate_optimal_position_size(
            balance=balance['USD'],
            price=price,
            risk_percentage=risk_percentage,
            volatility=atr / price,
            signal_strength=abs(combined_signal),
            market_condition=market_condition
        )

        # Risk management checks
        if not is_risk_acceptable(position_size, price, stop_loss, balance['USD']):
            return TradeResult(False, balance, current_position, "Risk parameters exceeded")

        # Market condition adjustments
        if should_adjust_for_market_condition(market_condition, volatility, combined_signal):
            position_size *= 0.8  # Reduce position size in unfavorable conditions
            logger.info(f"[{request_id}] Position size adjusted for market condition")

        # Execute the trade with the calculated parameters
        if action == 'BUY':
            return handle_buy_trade(
                price=price,
                position_size=position_size,
                balance=balance,
                current_position=current_position,
                stop_loss=calculate_dynamic_stop_loss(price, atr, volatility, 'BUY'),
                take_profit=calculate_dynamic_take_profit(price, atr, volatility, 'BUY'),
                request_id=request_id,
                performance_tracker=performance_tracker
            )
        elif action == 'SELL':
            return handle_sell_trade(
                price=price,
                position_size=position_size,
                balance=balance,
                current_position=current_position,
                market_condition=market_condition,
                volatility=volatility,
                request_id=request_id,
                performance_tracker=performance_tracker
            )
        
        return TradeResult(False, balance, current_position, "Invalid action")
        
    except Exception as e:
        logger.error(f"[{request_id}] Trade execution error: {str(e)}", exc_info=True)
        return TradeResult(False, balance, current_position, str(e))

def validate_trade_conditions(price: float, atr: float, market_condition: str, volatility: float, balance: Dict[str, float]) -> bool:
    """Validate all trade conditions"""
    return all([
        price > 0,
        atr > 0,
        market_condition in ['trending', 'ranging', 'volatile_trending', 'volatile_ranging'],
        volatility > 0,
        balance['USD'] > 0
    ])

def calculate_optimal_position_size(balance: float, price: float, risk_percentage: float, 
                                 volatility: float, signal_strength: float, market_condition: str) -> float:
    """Calculate optimal position size with market-adaptive sizing"""
    # Base position size
    base_size = (balance * risk_percentage) / (price * volatility)
    
    # Market condition multipliers - adjusted for ranging markets
    condition_multipliers = {
        'trending': 1.2,
        'volatile_trending': 1.0,
        'ranging': 1.0,  # Increased from 0.8 for ranging markets
        'volatile_ranging': 0.7
    }
    market_multiplier = condition_multipliers.get(market_condition, 0.8)
    
    # More aggressive signal strength multiplier for ranging markets
    if market_condition == 'ranging':
        signal_multiplier = min(2.0, max(0.5, signal_strength * 3))  # More aggressive in ranging
    else:
        signal_multiplier = min(1.5, max(0.5, signal_strength * 2))
    
    # Calculate final position size with limits
    position_size = base_size * market_multiplier * signal_multiplier
    max_position = balance * 0.15 / price  # Reduced from 0.2 for better risk management
    
    return min(position_size, max_position)

def is_risk_acceptable(position_size: float, price: float, stop_loss: float, balance: float) -> bool:
    """Enhanced risk assessment with market-adaptive limits"""
    potential_loss = position_size * abs(price - stop_loss)
    max_acceptable_loss = balance * 0.015  # Reduced from 0.02 for tighter risk control
    return potential_loss <= max_acceptable_loss

def should_adjust_for_market_condition(market_condition: str, volatility: float, signal_strength: float) -> bool:
    """Determine if position size should be adjusted based on market conditions"""
    return (
        market_condition in ['volatile_ranging', 'volatile_trending'] or
        volatility > 0.025 or  # Reduced from 0.03 for earlier volatility detection
        abs(signal_strength) < 0.3  # Reduced from 0.5 for more trading opportunities
    )

def calculate_dynamic_stop_loss(price: float, atr: float, volatility: float, action: str) -> float:
    """Calculate dynamic stop loss with tighter ranges for ranging markets"""
    # Tighter stops in ranging markets
    base_distance = atr * (1.2 + volatility * 4)  # Reduced multipliers for tighter stops
    return price - base_distance if action == 'BUY' else price + base_distance

def calculate_dynamic_take_profit(price: float, atr: float, volatility: float, action: str) -> float:
    """Calculate dynamic take profit with market-adaptive targets"""
    # Adjusted for quicker profits in ranging markets
    base_distance = atr * (2.0 + volatility * 5)  # Reduced for quicker profits
    return price + base_distance if action == 'BUY' else price - base_distance

def calculate_position_size(balance, current_price, risk_percentage, volatility):
    try:
        risk_amount = balance * risk_percentage
        position_size = risk_amount / (current_price * volatility)
        max_position_size = balance / current_price  # Maximum position size
        return min(position_size, max_position_size, balance * 0.1)  # Limit to 10% of balance
    except Exception as e:
        logger.error(f"Error calculating position size: {str(e)}")
        raise TradeExecutionError(f"Failed to calculate position size: {str(e)}")

def calculate_dynamic_levels(current_price, atr, volatility):
    try:
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
    except Exception as e:
        logger.error(f"Error calculating dynamic levels: {str(e)}")
        raise TradeExecutionError(f"Failed to calculate dynamic levels: {str(e)}")

def log_trade_execution(action, quantity, price, profit_loss=None, total_profit_loss=None, request_id=None, market_condition=None, volatility=None):
    try:
        message = f"[{request_id}] TRADE_EXECUTION: {action} {quantity:.6f} BTC @ {price:.2f} USD"
        if profit_loss is not None:
            message += f", Profit/Loss: {profit_loss:.2f}%"
        if total_profit_loss is not None:
            message += f", Total Profit/Loss: {total_profit_loss:.2f} USD"
        message += f", Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        if market_condition is not None:
            message += f", Market Condition: {market_condition}"
        if volatility is not None:
            message += f", Volatility: {volatility:.6f}"
        logger.info(message)
    except Exception as e:
        logger.error(f"Error logging trade execution: {str(e)}")

def check_stop_loss_take_profit(current_price, atr, volatility, market_condition, request_id, balance, current_position, performance_tracker):
    try:
        if current_position['side'] != 'long':
            return balance, current_position

        profit_loss_percentage = (current_price - current_position['average_price']) / current_position['average_price'] * 100

        # Track highest price for trailing stop
        if current_price > current_position.get('highest_price', current_price):
            current_position['highest_price'] = current_price
            
            # Dynamic trailing stop based on profit level
            if profit_loss_percentage >= 5.0:
                trail_percentage = 0.02  # 2% trail at 5% profit
            elif profit_loss_percentage >= 8.0:
                trail_percentage = 0.015  # 1.5% trail at 8% profit
            elif profit_loss_percentage >= 10.0:
                trail_percentage = 0.01  # 1% trail at 10% profit
            else:
                trail_percentage = 0.03  # 3% trail below 5% profit
                
            # Calculate new stop loss based on trailing percentage
            new_stop_loss = current_price * (1 - trail_percentage)
            
            # Update stop loss if it's higher than current
            if new_stop_loss > current_position['stop_loss']:
                current_position['stop_loss'] = new_stop_loss
                logger.info(f"[{request_id}] Trailing stop-loss updated: {new_stop_loss:.2f}")

        # Dynamic take-profit adjustment
        new_take_profit = calculate_dynamic_take_profit(
            current_price, 
            current_position['average_price'],
            atr,
            volatility,
            market_condition
        )
        current_position['take_profit'] = new_take_profit

        # Check for exit conditions
        if current_price <= current_position['stop_loss'] or current_price >= current_position['take_profit']:
            return execute_trade(
                'SELL',
                current_price,
                atr,
                -1.0,  # Strong sell signal
                ["Stop-loss/Take-profit triggered"],
                current_position['stop_loss'],
                current_position['take_profit'],
                market_condition,
                volatility,
                request_id,
                balance,
                current_position,
                0.01,  # Risk percentage
                performance_tracker
            )

    except Exception as e:
        logger.error(f"[{request_id}] Error in stop-loss/take-profit check: {e}", exc_info=True)
        raise TradeExecutionError(f"Failed to check stop loss / take profit: {str(e)}")

    return balance, current_position

def calculate_dynamic_take_profit(current_price: float, entry_price: float, atr: float, volatility: float, market_condition: str) -> float:
    """
    Calculate dynamic take-profit level based on market conditions
    """
    try:
        # Base take-profit multiplier
        base_tp_multiplier = 2.0
        
        # Adjust based on volatility
        if volatility > 0.02:  # High volatility
            base_tp_multiplier *= 1.3
        elif volatility < 0.005:  # Low volatility
            base_tp_multiplier *= 0.8
            
        # Adjust based on market condition
        if 'bullish' in market_condition:
            base_tp_multiplier *= 1.2
        elif 'bearish' in market_condition:
            base_tp_multiplier *= 0.8
            
        # Calculate take-profit level
        profit_target = atr * base_tp_multiplier
        take_profit = current_price + profit_target
        
        # Ensure minimum profit target
        min_profit = entry_price * 1.01  # Minimum 1% profit
        return max(take_profit, min_profit)
        
    except Exception as e:
        logger.error(f"Error calculating dynamic take-profit: {str(e)}")
        return current_price * 1.02  # Default to 2% profit target on error

def calculate_sell_ratio(profit_loss: float, market_condition: str, volatility: float) -> float:
    """
    Calculate the optimal sell ratio based on profit level and market conditions
    Returns a value between 0.3 (30%) and 1.0 (100%) of the position
    """
    try:
        # Base ratio starts at 30%
        base_ratio = 0.3
        
        # Adjust based on profit level
        if profit_loss >= 5.0:
            base_ratio = 0.5  # 50% at 5% profit
        elif profit_loss >= 8.0:
            base_ratio = 0.7  # 70% at 8% profit
        elif profit_loss >= 10.0:
            base_ratio = 1.0  # Full position at 10% profit
            
        # Adjust based on market condition
        if 'high_volatility' in market_condition:
            base_ratio *= 1.2  # More aggressive in high volatility
        elif 'bearish' in market_condition:
            base_ratio *= 1.3  # More aggressive in bearish conditions
            
        # Adjust based on volatility
        if volatility > 0.02:  # High volatility
            base_ratio *= 1.2
        elif volatility < 0.005:  # Low volatility
            base_ratio *= 0.8
            
        # Ensure ratio stays between 0.3 and 1.0
        return min(max(base_ratio, 0.3), 1.0)
    except Exception as e:
        logger.error(f"Error calculating sell ratio: {str(e)}")
        return 1.0  # Default to full position sell on error

def handle_buy_trade(price, position_size, balance, current_position, stop_loss, take_profit, request_id, performance_tracker):
    """Handle buy trade execution"""
    try:
        cost = price * position_size
        if cost > balance['USD']:
            raise InsufficientFundsError("Insufficient USD balance for trade")

        balance['USD'] -= cost
        balance['BTC'] += position_size

        current_position.update({
            'side': 'long',
            'total_quantity': position_size,
            'average_price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'highest_price': price,
            'last_update': datetime.now()
        })

        log_trade_execution('BUY', position_size, price, request_id=request_id)
        return TradeResult(True, balance, current_position, "Buy trade executed successfully")

    except Exception as e:
        logger.error(f"[{request_id}] Buy trade error: {str(e)}", exc_info=True)
        return TradeResult(False, balance, current_position, str(e))

def handle_sell_trade(price, position_size, balance, current_position, market_condition, volatility, request_id, performance_tracker):
    """Handle sell trade execution"""
    try:
        if current_position['side'] != 'long' or current_position['total_quantity'] <= 0:
            return TradeResult(False, balance, current_position, "No position to sell")

        sell_amount = min(position_size, current_position['total_quantity'])
        proceeds = price * sell_amount

        profit_loss = ((price - current_position['average_price']) / current_position['average_price']) * 100
        
        balance['USD'] += proceeds
        balance['BTC'] -= sell_amount
        
        if sell_amount >= current_position['total_quantity']:
            current_position.update({
                'side': None,
                'total_quantity': 0,
                'average_price': 0,
                'stop_loss': None,
                'take_profit': None,
                'highest_price': None,
                'last_update': datetime.now()
            })
        else:
            current_position['total_quantity'] -= sell_amount
            current_position['last_update'] = datetime.now()

        log_trade_execution('SELL', sell_amount, price, profit_loss, request_id=request_id, 
                          market_condition=market_condition, volatility=volatility)
        
        if hasattr(performance_tracker, 'update_trade_performance'):
            performance_tracker.update_trade_performance(profit_loss)

        return TradeResult(True, balance, current_position, f"Sell trade executed successfully. P/L: {profit_loss:.2f}%")

    except Exception as e:
        logger.error(f"[{request_id}] Sell trade error: {str(e)}", exc_info=True)
        return TradeResult(False, balance, current_position, str(e))

