import time
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import time
from datetime import datetime
from colorama import Fore, Style
import logging

logger = logging.getLogger(__name__)

console = Console()

def print_boxed(message: str, color: str = Fore.WHITE):
    """Print a message in a box"""
    width = len(message) + 2
    border = "─" * width
    console.print(f"{color}╭{border}╮")
    console.print(f"│ {message} │")
    console.print(f"╰{border}╯{Style.RESET_ALL}")

def format_market_structure(market_structure: dict) -> str:
    """Format market structure dictionary into a readable string"""
    if not isinstance(market_structure, dict):
        return str(market_structure)
    
    try:
        trend_info = market_structure.get('trend', {})
        vol_info = market_structure.get('volatility', {})
        volume_info = market_structure.get('volume', {})
        
        trend_str = f"Trend: {trend_info.get('direction', 'unknown')} ({trend_info.get('strength', 0):.4f})"
        vol_str = f"Vol: {vol_info.get('description', 'unknown')}"
        volume_str = f"Volume: {volume_info.get('description', 'low')}"
        
        return f"{trend_str}, {vol_str}, {volume_str}"
    except Exception as e:
        logger.error(f"Error formatting market structure: {e}")
        return str(market_structure)

def print_market_status(current_price: float, atr: float, signal: float, 
                       market_condition: str, market_structure: dict, news_sentiment: float = None):
    """Print current market status in a formatted table"""
    market_table = Table(show_header=False, box=box.ROUNDED)
    market_table.add_column("Metric", style="cyan")
    market_table.add_column("Value", style="yellow")
    
    market_table.add_row("Current Price", f"${current_price:,.2f}")
    market_table.add_row("ATR", f"{atr:.4f}")
    
    signal_color = "green" if signal > 0 else "red" if signal < 0 else "yellow"
    market_table.add_row("Signal", f"[{signal_color}]{signal:+.4f}[/]")
    
    if news_sentiment is not None:
        sentiment_color = "green" if news_sentiment > 0 else "red" if news_sentiment < 0 else "yellow"
        market_table.add_row("News Sentiment", f"[{sentiment_color}]{news_sentiment:+.4f}[/]")
    
    market_table.add_row("Market Condition", market_condition)
    market_table.add_row("Market Structure", format_market_structure(market_structure))
    
    console.print(Panel(market_table, title="Market Status", border_style="blue"))

def print_balance(balance: dict, current_price: float):
    """Print current balance information"""
    balance_table = Table(show_header=False, box=box.ROUNDED)
    balance_table.add_column("Asset", style="cyan")
    balance_table.add_column("Amount", style="yellow")
    
    usd_balance = balance.get('USD', 0)
    btc_balance = balance.get('BTC', 0)
    total_usd = usd_balance + (btc_balance * current_price)
    
    balance_table.add_row("USD Balance", f"${usd_balance:,.2f}")
    balance_table.add_row("BTC Balance", f"{btc_balance:.8f} BTC")
    balance_table.add_row("Total (USD)", f"${total_usd:,.2f}")
    
    console.print(Panel(balance_table, title="Balance Information", border_style="green"))

def format_position(position: dict, current_price: float) -> Table:
    """Format position information into a table"""
    position_table = Table(show_header=False, box=box.ROUNDED)
    position_table.add_column("Metric", style="cyan")
    position_table.add_column("Value", style="yellow")
    
    side = position.get('side')
    if not side:
        position_table.add_row("Position", "No active position")
        return position_table
    
    avg_price = position.get('average_price', 0)
    quantity = position.get('total_quantity', 0)
    stop_loss = position.get('stop_loss')
    take_profit = position.get('take_profit')
    
    # Calculate profit/loss if position exists
    if avg_price and quantity:
        pnl = ((current_price - avg_price) / avg_price) * 100
        pnl_color = "green" if pnl > 0 else "red"
        position_table.add_row("Side", side.upper())
        position_table.add_row("Quantity", f"{quantity:.8f} BTC")
        position_table.add_row("Average Price", f"${avg_price:,.2f}")
        position_table.add_row("P/L", f"[{pnl_color}]{pnl:+.2f}%[/]")
        
        if stop_loss:
            position_table.add_row("Stop Loss", f"${stop_loss:,.2f}")
        if take_profit:
            position_table.add_row("Take Profit", f"${take_profit:,.2f}")
    
    return position_table

def print_position(position: dict, current_price: float):
    """Print current position information"""
    position_table = format_position(position, current_price)
    console.print(Panel(position_table, title="Position Information", border_style="magenta"))

def print_next_check_info(next_check_time: datetime, wait_time: int):
    """Print information about next check time"""
    time_table = Table(show_header=False, box=box.ROUNDED)
    time_table.add_column("Metric", style="cyan")
    time_table.add_column("Value", style="yellow")
    
    time_table.add_row("Next Check", next_check_time.strftime("%Y-%m-%d %H:%M:%S"))
    time_table.add_row("Wait Time", f"{wait_time} seconds")
    
    console.print(Panel(time_table, title="Next Check Information", border_style="blue"))

def wait_with_progress(seconds: int):
    """Show a progress bar while waiting"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Waiting for next check...", total=seconds)
        for _ in range(seconds):
            time.sleep(1)
            progress.advance(task)

