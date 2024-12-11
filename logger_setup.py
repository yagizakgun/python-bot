import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from colorama import Fore, Back, Style, init

# Initialize colorama
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        levelname = record.levelname
        msg = super().format(record)
        if sys.stdout.isatty():  # Check if output is terminal
            msg = f"{self.COLORS.get(levelname, Fore.RESET)}{msg}{Style.RESET_ALL}"
        return msg

def setup_logger(log_level='DEBUG'):
    logger = logging.getLogger("TradingBot")
    logger.setLevel(logging.DEBUG)  # Set base level to DEBUG

    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # File handler (with rotation)
    log_file = f"logs/trading_bot_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    file_formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Convert string log level to logging constant for console handler
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    console_handler.setLevel(numeric_level)

    # Add handlers if they don't exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
