class TradingBotError(Exception):
    """Base class for Trading Bot exceptions"""
    pass

class APIError(TradingBotError):
    """Raised when there's an error with API calls"""
    pass

class DataFetchError(TradingBotError):
    """Raised when there's an error fetching data"""
    pass

class SignalGenerationError(TradingBotError):
    """Raised when there's an error generating trading signals"""
    pass

class TradeExecutionError(TradingBotError):
    """Raised when there's an error executing a trade"""
    pass

class ConfigurationError(TradingBotError):
    """Raised when there's an error in the configuration"""
    pass

class InsufficientFundsError(TradingBotError):
    """Raised when there are insufficient funds for a trade"""
    pass

