import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Bot configuration
SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
TIMEFRAMES = os.getenv('TIMEFRAMES', '1m,3m,5m,15m,1h,4h').split(',')
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '10000'))
WAIT_TIME = int(os.getenv('WAIT_TIME', '60'))  # 1 dakika (varsayılan)

# Gradual entry parameters
NUM_ENTRIES = int(os.getenv('NUM_ENTRIES', '3'))
ENTRY_INTERVAL = int(os.getenv('ENTRY_INTERVAL', '120'))
ENTRY_PERCENTAGE = [float(x) for x in os.getenv('ENTRY_PERCENTAGE', '0.4,0.3,0.3').split(',')]

# Risk management parameters
RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '0.01'))
MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', '0.1'))

# Cooling-off period (in minutes)
COOLING_OFF_PERIOD = int(os.getenv('COOLING_OFF_PERIOD', '30'))

# Binance API configuration
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# News Analyzer Configuration
NEWS_CONFIG = {
    "enabled": True,  # Disabled by default
    "cryptocompare_api_key": "d96372f92c73a009488ca43e7a8210c940f819e475fbe4cca64a3a33552ca666",  # Add your API key
    "cryptopanic_api_key": "652ffde18cdefec15d909c11c4221379be474aec",    # Add your API key
    "cache_duration": 300,        # 5 minutes
    "sentiment_threshold": 0.5,
    "emergency_threshold": -0.8,
    "keywords": ["bitcoin", "btc", "crypto", "cryptocurrency"],
    "max_news_age": 3600,        # 1 hour
    "update_interval": 60         # 1 minute
}

