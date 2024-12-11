import time
import logging
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import colorama
from colorama import Fore, Style
import asyncio

# Local imports
from config import *
from data_fetcher import get_historical_data, get_current_price
from signal_generator import get_market_analysis, analyze_multiple_timeframes, combine_signals, calculate_current_volatility
from trade_executor import execute_trade, check_stop_loss_take_profit
from utils import (generate_request_id, log_to_json, log_to_csv, calculate_dynamic_levels,
                  calculate_dynamic_cooling_period, can_trade)
from performance_tracker import PerformanceTracker
from logger_setup import setup_logger
from exceptions import *
from logger_utils import *
from indicators import calculate_indicators
from news_analyzer import NewsSentimentAnalyzer, NewsConfig

# Initialize colorama and logger
colorama.init(autoreset=True)
logger = setup_logger()

def initialize_trading_system():
    """Initialize all trading components"""
    try:
        client = Client(API_KEY, API_SECRET)
        performance_tracker = PerformanceTracker()
        balance = {'USD': INITIAL_BALANCE, 'BTC': 0}
        current_position = {
            'side': None,
            'entries': [],
            'total_quantity': 0,
            'average_price': 0,
            'last_update': None,
            'stop_loss': None,
            'take_profit': None,
            'highest_price': None
        }
        
        # Initialize news analyzer
        news_config = NewsConfig(**NEWS_CONFIG)
        news_analyzer = NewsSentimentAnalyzer(news_config)
        
        return client, performance_tracker, balance, current_position, news_analyzer
    except BinanceAPIException as e:
        logger.critical(f"Failed to initialize Binance client: {str(e)}")
        raise ConfigurationError("Invalid API credentials") from e

async def analyze_market_conditions(client, config, request_id, news_analyzer):
    """Analyze current market conditions and generate signals"""
    try:
        # Get market signals
        signals, market_condition = await analyze_multiple_timeframes(
            client, SYMBOL, TIMEFRAMES, config
        )
        
        # Get current price
        current_price = await get_current_price(client, SYMBOL)
        if current_price is None:
            raise DataFetchError("Could not fetch current price")

        # Get hourly data and indicators
        hourly_data = await get_historical_data(client, SYMBOL, '1h')
        if hourly_data is not None and not hourly_data.empty:
            hourly_data = calculate_indicators(hourly_data, '1h')
            atr = hourly_data['ATR'].iloc[-1]
        else:
            atr = current_price * 0.01

        # Get market analysis
        market_analysis = get_market_analysis(hourly_data, config)
        market_structure = market_analysis['structure']
        if market_condition is None:
            market_condition = market_analysis['condition']

        # Calculate volatility and signals
        volatility = calculate_current_volatility(hourly_data)
        
        # Update news sentiment if needed
        if news_analyzer.should_update():
            await news_analyzer.update_sentiment()
        
        # Get base signals
        combined_signal, buy_threshold, sell_threshold = combine_signals(
            signals, volatility, market_condition
        )
        
        # Adjust signal based on news sentiment
        combined_signal = news_analyzer.adjust_signal(combined_signal)

        # Calculate stop loss and take profit
        stop_loss, take_profit = calculate_dynamic_levels(current_price, atr, volatility)

        return {
            'signals': signals,
            'current_price': current_price,
            'market_condition': market_condition,
            'market_structure': market_structure,
            'volatility': volatility,
            'atr': atr,
            'combined_signal': combined_signal,
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'news_sentiment': news_analyzer.current_sentiment if news_analyzer.config.enabled else 0.0,
            'news_emergency': news_analyzer.emergency_signal if news_analyzer.config.enabled else False
        }
    except Exception as e:
        logger.error(f"Market analysis failed: {str(e)}")
        raise

async def run_trading_iteration(client, balance, current_position, last_trade_time, 
                              performance_tracker, config, request_id, news_analyzer):
    """Run a single trading iteration"""
    try:
        # Analyze market
        market_data = await analyze_market_conditions(client, config, request_id, news_analyzer)
        
        # Check for news emergencies
        if market_data['news_emergency']:
            logger.warning("Emergency news signal detected - forcing risk reduction")
            # Implement emergency risk reduction here
            
        # Process signals and execute trades
        balance, current_position, last_trade_time = process_trading_signals(
            market_data, balance, current_position, last_trade_time,
            performance_tracker, request_id
        )
        
        # Update displays
        print_market_status(
            market_data['current_price'],
            market_data['atr'],
            market_data['combined_signal'],
            market_data['market_condition'],
            market_data['market_structure'],
            market_data['news_sentiment'] if news_analyzer.config.enabled else None
        )
        print_balance(balance, market_data['current_price'])
        print_position(current_position, market_data['current_price'])
        
        # Log data
        log_trading_data(market_data, balance, current_position, request_id)
        
        return balance, current_position, last_trade_time
    
    except Exception as e:
        logger.error(f"Error in trading iteration: {str(e)}")
        raise

async def run_bot():
    """Main bot execution loop"""
    console.print(Panel("Bot starting...", title="Trading Bot Starting", border_style="green"))

    try:
        # Initialize trading system
        client, performance_tracker, balance, current_position, news_analyzer = initialize_trading_system()
        last_trade_time = None
        
        # Initialize news analyzer
        await news_analyzer.initialize()
        news_analyzer.load_state()  # Load previous state if available
        
        # Trading configuration
        config = {
            'volatility_window': 20,
            'trend_window': 50,
            'high_volatility_threshold': 0.02,
            'low_volatility_threshold': 0.005,
            'strong_trend_threshold': 0.1,
            'bb_width_threshold': 0.05
        }

        while True:
            try:
                request_id = generate_request_id()
    
                console.print(Panel("", title=f"Starting Iteration (ID: {request_id})", border_style="cyan"))

                # Run trading iteration
                balance, current_position, last_trade_time = await run_trading_iteration(
                    client, balance, current_position, last_trade_time,
                    performance_tracker, config, request_id, news_analyzer
                )

                # Wait for next iteration
                next_check_time = datetime.now() + timedelta(seconds=WAIT_TIME)
                logger.info(f"Next check time: {next_check_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print_next_check_info(next_check_time, WAIT_TIME)
                wait_with_progress(WAIT_TIME)
                
                console.print(Panel("", title=f"Iteration Complete (ID: {request_id})", border_style="magenta"))

            except (BinanceAPIException, ConnectionError) as e:
                logger.error(f"API error (ID: {request_id}): {str(e)}")
                time.sleep(60)
            except Exception as e:
                logger.error(f"Unexpected error (ID: {request_id}): {str(e)}", exc_info=True)
                time.sleep(60)

    except Exception as e:
        logger.critical(f"Critical error in main loop: {str(e)}", exc_info=True)
        raise
    finally:
        # Clean up
        if news_analyzer:
            news_analyzer.save_state()
            await news_analyzer.close()

def process_trading_signals(market_data, balance, current_position, last_trade_time, 
                          performance_tracker, request_id):
    """Process trading signals and execute trades if conditions are met"""
    try:
        # Get last trade profit if available
        last_trade_profit = None
        if hasattr(performance_tracker, 'last_trade_profit'):
            last_trade_profit = performance_tracker.last_trade_profit

        combined_signal = market_data['combined_signal']
        current_price = market_data['current_price']
        
        # Check if we can trade
        can_trade_now, reason = can_trade(
            last_trade_time,
            market_data['market_condition'],
            market_data['volatility'],
            COOLING_OFF_PERIOD,
            signal_strength=combined_signal,
            last_trade_profit=last_trade_profit
        )
        
        logger.info(f"Trade status: {reason}")
        
        if not can_trade_now:
            return balance, current_position, last_trade_time

        # Generate trade reasons
        reasons = [
            f"Signal strength: {combined_signal:.2f}",
            f"Market condition: {market_data['market_condition']}",
            f"Volatility: {market_data['volatility']:.4f}",
            f"Market structure: {market_data['market_structure']}"
        ]

        # Check signal strength
        if abs(combined_signal) > max(market_data['buy_threshold'], abs(market_data['sell_threshold'])):
            if combined_signal > market_data['buy_threshold'] and current_position['side'] != 'long':
                logger.info("BUY signal detected")
                trade_result = execute_trade(
                    'BUY', current_price, market_data['atr'],
                    combined_signal, reasons,
                    market_data['stop_loss'], market_data['take_profit'],
                    market_data['market_condition'], market_data['volatility'],
                    request_id, balance, current_position,
                    RISK_PERCENTAGE, performance_tracker
                )
                if trade_result.success:
                    balance = trade_result.balance
                    current_position = trade_result.position
                    last_trade_time = datetime.now()
                else:
                    logger.error(f"Trade execution failed: {trade_result.message}")
            
            elif combined_signal < market_data['sell_threshold'] and current_position['side'] == 'long':
                logger.info("SELL signal detected")
                trade_result = execute_trade(
                    'SELL', current_price, market_data['atr'],
                    combined_signal, reasons,
                    market_data['stop_loss'], market_data['take_profit'],
                    market_data['market_condition'], market_data['volatility'],
                    request_id, balance, current_position,
                    RISK_PERCENTAGE, performance_tracker
                )
                if trade_result.success:
                    balance = trade_result.balance
                    current_position = trade_result.position
                    last_trade_time = datetime.now()
                else:
                    logger.error(f"Trade execution failed: {trade_result.message}")
        else:
            logger.info(f"Signal not strong enough. No trade executed. (Signal: {combined_signal:.2f})")

        return balance, current_position, last_trade_time
    
    except Exception as e:
        logger.error(f"Error processing trading signals: {str(e)}")
        raise

def log_trading_data(market_data, balance, current_position, request_id):
    """Log trading data to JSON and CSV"""
    try:
        current_price = market_data['current_price']
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "current_price": current_price,
            "combined_signal": market_data['combined_signal'],
            "market_condition": market_data['market_condition'],
            "market_structure": market_data['market_structure'],
            "volatility": market_data['volatility'],
            "atr": market_data['atr'],
            "balance_usd": balance['USD'],
            "balance_btc": balance['BTC'],
            "total_balance_usd": balance['USD'] + balance['BTC'] * current_price,
            "position": current_position,
            "cooling_period_info": {
                "dynamic_period": calculate_dynamic_cooling_period(
                    market_data['market_condition'],
                    market_data['volatility'],
                    COOLING_OFF_PERIOD,
                    market_data['combined_signal']
                ),
                "market_condition": market_data['market_condition'],
                "volatility": market_data['volatility']
            }
        }
        log_to_json(log_data)
        log_to_csv(log_data)
    except Exception as e:
        logger.error(f"Error logging trading data: {str(e)}")

if __name__ == "__main__":
    try:
        asyncio.run(run_bot())
    except ConfigurationError as e:
        logger.critical(f"Configuration error: {str(e)}")
        print_boxed("Bot failed to start. Please check your configuration.", Fore.RED)
    except Exception as e:
        logger.critical(f"Unexpected error during bot startup: {str(e)}", exc_info=True)
        print_boxed("Bot failed to start due to an unexpected error.", Fore.RED)

