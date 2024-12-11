import numpy as np
from datetime import datetime, timedelta

class PerformanceTracker:
    def __init__(self):
        self.trades = []
        self.daily_returns = []
        self.risk_free_rate = 0.02  # Yıllık %2 risk-free rate varsayalım

    def log_trade(self, action, price, quantity, amount, profit_loss=None):
        self.trades.append((datetime.now(), action, price, quantity, amount, profit_loss))

    def get_recent_trades(self, minutes=5):
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [trade for trade in self.trades if trade[0] > cutoff_time]

    def get_total_profit_loss(self):
        return sum(trade[5] for trade in self.trades if trade[5] is not None)

    def get_total_profit_loss_amount(self):
        return sum((trade[2] * trade[3] - trade[4]) for trade in self.trades if trade[1] == 'SELL')

    def calculate_daily_returns(self, initial_balance, current_balance):
        if len(self.daily_returns) == 0 or (datetime.now().date() > self.daily_returns[-1][0].date()):
            daily_return = (current_balance - initial_balance) / initial_balance
            self.daily_returns.append((datetime.now(), daily_return))

    def calculate_sharpe_ratio(self):
        if len(self.daily_returns) < 2:
            return None
        returns = [r[1] for r in self.daily_returns]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return None
        return (avg_return - (self.risk_free_rate / 365)) / std_return * np.sqrt(365)

    def calculate_max_drawdown(self):
        if len(self.daily_returns) < 2:
            return None
        cumulative_returns = np.cumprod(1 + np.array([r[1] for r in self.daily_returns]))
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (peak - cumulative_returns) / peak
        return np.max(drawdown)

    def calculate_win_loss_ratio(self):
        profitable_trades = sum(1 for trade in self.trades if trade[5] is not None and trade[5] > 0)
        losing_trades = sum(1 for trade in self.trades if trade[5] is not None and trade[5] <= 0)
        if losing_trades == 0:
            return None
        return profitable_trades / losing_trades if losing_trades > 0 else None

    def get_performance_metrics(self, initial_balance, current_balance):
        self.calculate_daily_returns(initial_balance, current_balance)
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()
        win_loss_ratio = self.calculate_win_loss_ratio()

        return {
            "total_trades": len(self.trades),
            "profitable_trades": sum(1 for trade in self.trades if trade[5] is not None and trade[5] > 0),
            "total_profit_loss": self.get_total_profit_loss(),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_loss_ratio": win_loss_ratio
        }

