from datetime import datetime, timedelta
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()

def generate_concise_summary(current_price, combined_signal, market_condition, market_structure, balance, current_position, performance_tracker, request_id):
    summary_table = Table(title="Market Summary", box=box.DOUBLE_EDGE)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Price", f"{current_price:.2f}")
    summary_table.add_row("Signal", f"{combined_signal:.2f}")
    summary_table.add_row("Market Condition", market_condition)
    summary_table.add_row("Market Structure", market_structure)
    
    total_balance = balance['USD'] + balance['BTC'] * current_price
    profit_loss = performance_tracker.get_total_profit_loss_amount()
    
    summary_table.add_row("Total Balance", f"{total_balance:.2f} USD")
    summary_table.add_row("Profit/Loss", f"{profit_loss:.2f} USD")
    
    if current_position['side'] == 'long':
        current_profit_loss = (current_price - current_position['average_price']) / current_position['average_price'] * 100
        summary_table.add_row("Current Position", f"{current_position['total_quantity']:.6f} BTC")
        summary_table.add_row("Position P/L", f"{current_profit_loss:.2f}%")
    
    console.print(Panel(summary_table, title=f"[bold green]Concise Summary (ID: {request_id})[/bold green]", expand=False))

def generate_short_term_summary(interval_minutes, performance_tracker, request_id):
    current_time = datetime.now()
    start_time = current_time - timedelta(minutes=interval_minutes)
    console.print(f"[cyan]=== {interval_minutes}-Minute Summary (ID: {request_id}) ===")
    recent_trades = performance_tracker.get_recent_trades(minutes=interval_minutes)
    
    if recent_trades:
        total_trades = len(recent_trades)
        profitable_trades = sum(1 for trade in recent_trades if trade[5] is not None and trade[5] > 0)
        total_profit_loss = sum(trade[5] for trade in recent_trades if trade[5] is not None)
        
        summary_table = Table(title=f"Last {interval_minutes} Minutes Trading Summary", box=box.SIMPLE_HEAVY)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Total Trades", str(total_trades))
        summary_table.add_row("Profitable Trades", str(profitable_trades))
        summary_table.add_row("Total Profit/Loss", f"{total_profit_loss:.2f} USD")
        
        console.print(summary_table)
    else:
        console.print(f"[yellow]No trades in the last {interval_minutes} minutes.")

def generate_daily_summary(balance, performance_tracker, symbol, get_current_price, client, request_id):
    console.print(f"[cyan]=== Daily Summary (ID: {request_id}) ===")
    total_balance = balance['USD'] + balance['BTC'] * get_current_price(client, symbol)
    
    summary_table = Table(title="Daily Trading Summary", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    summary_table.add_row("Total Balance", f"{total_balance:.2f} USD")
    summary_table.add_row("Total Profit/Loss", f"{performance_tracker.get_total_profit_loss():.2f} USD")
    summary_table.add_row("Total Trades", str(len(performance_tracker.trades)))
    
    console.print(summary_table)

def log_daily_trade_summary(performance_tracker, request_id):
    console.print(f"[cyan]=== Daily Trade Summary (ID: {request_id}) ===")
    
    summary_table = Table(title="Daily Trade Details", box=box.SIMPLE_HEAVY)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="magenta")
    
    total_trades = len(performance_tracker.trades)
    buy_trades = [trade for trade in performance_tracker.trades if trade[1] == 'BUY']
    sell_trades = [trade for trade in performance_tracker.trades if trade[1] == 'SELL']
    
    summary_table.add_row("Total Trades", str(total_trades))
    summary_table.add_row("Buy Trades", str(len(buy_trades)))
    summary_table.add_row("Sell Trades", str(len(sell_trades)))
    summary_table.add_row("Total Profit/Loss", f"{performance_tracker.get_total_profit_loss():.2f} USD")
    
    console.print(summary_table)

