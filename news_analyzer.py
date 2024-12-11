import asyncio
import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
import aiohttp
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from rich.console import Console
from rich.panel import Panel
from rich import box

console = Console()
logger = logging.getLogger(__name__)

@dataclass
class NewsConfig:
    """Configuration for news analysis"""
    enabled: bool = False
    cryptocompare_api_key: str = ""
    cryptopanic_api_key: str = ""
    cache_duration: int = 300  # 5 minutes
    sentiment_threshold: float = 0.5
    emergency_threshold: float = -0.8
    keywords: List[str] = None
    max_news_age: int = 3600  # 1 hour
    update_interval: int = 60  # 1 minute

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = ["bitcoin", "btc", "crypto", "cryptocurrency"]

class NewsCache:
    """Cache for news data"""
    def __init__(self, cache_duration: int):
        self.cache: Dict[str, Dict] = {}
        self.cache_duration = cache_duration

    def get(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp <= self.cache_duration:
                return data
            del self.cache[key]
        return None

    def set(self, key: str, data: Dict):
        self.cache[key] = (data, time.time())

    def clear_old_entries(self):
        current_time = time.time()
        self.cache = {
            k: v for k, v in self.cache.items()
            if current_time - v[1] <= self.cache_duration
        }

class SentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Enhanced crypto-specific sentiment lexicon
        self.crypto_positive = {
            'bullish': 1.5, 'surge': 1.3, 'rally': 1.2, 'breakout': 1.4,
            'adoption': 1.2, 'institutional': 1.1, 'accumulation': 1.2,
            'upgrade': 1.1, 'partnership': 1.2, 'innovation': 1.1,
            'support': 0.8, 'gain': 1.0, 'growth': 0.9
        }
        
        self.crypto_negative = {
            'bearish': -1.5, 'crash': -1.5, 'dump': -1.3, 'ban': -1.4,
            'hack': -1.5, 'scam': -1.5, 'fraud': -1.5, 'manipulation': -1.3,
            'sell-off': -1.2, 'correction': -0.8, 'vulnerability': -1.1,
            'weakness': -0.9, 'rejection': -0.8, 'regulation': -0.7
        }
        
        # Market indicators
        self.market_indicators = {
            'overbought': -0.5, 'oversold': 0.5,
            'resistance': -0.3, 'support': 0.3,
            'volume': 0.1, 'momentum': 0.2
        }

    def preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        
        # Tokenize and lemmatize
        words = word_tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)

    def get_crypto_sentiment_score(self, text: str) -> float:
        """Get crypto-specific sentiment score"""
        words = word_tokenize(text.lower())
        score = 0
        word_count = 0
        
        for word in words:
            if word in self.crypto_positive:
                score += self.crypto_positive[word]
                word_count += 1
            elif word in self.crypto_negative:
                score += self.crypto_negative[word]
                word_count += 1
            elif word in self.market_indicators:
                score += self.market_indicators[word]
                word_count += 1
        
        return score / max(word_count, 1)

    def analyze(self, text: str) -> float:
        """Comprehensive sentiment analysis"""
        if not text:
            return 0.0
            
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Get base sentiment using TextBlob
        blob = TextBlob(processed_text)
        base_sentiment = blob.sentiment.polarity
        
        # Get crypto-specific sentiment
        crypto_sentiment = self.get_crypto_sentiment_score(processed_text)
        
        # Combine sentiments with weights
        # Give more weight to crypto-specific sentiment
        final_sentiment = (crypto_sentiment * 0.7) + (base_sentiment * 0.3)
        
        # Normalize to [-1, 1] range
        return max(min(final_sentiment, 1.0), -1.0)

class NewsSentimentAnalyzer:
    """Main class for news sentiment analysis"""
    def __init__(self, config: NewsConfig):
        self.config = config
        self.cache = NewsCache(config.cache_duration)
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update = datetime.min
        self.current_sentiment = 0.0
        self.emergency_signal = False
        console.print("[bold cyan]News Sentiment Analyzer initialized[/bold cyan]")
        
    async def initialize(self):
        """Initialize the analyzer"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            console.print("[green]News API session initialized successfully[/green]")

    async def close(self):
        """Close resources"""
        if self.session:
            await self.session.close()
            self.session = None
            console.print("[yellow]News API session closed[/yellow]")

    async def fetch_cryptocompare_news(self) -> List[Dict]:
        """Fetch news from CryptoCompare"""
        if not self.config.cryptocompare_api_key:
            console.print("[yellow]CryptoCompare API key not configured[/yellow]")
            return []

        try:
            console.print("[cyan]Fetching news from CryptoCompare...[/cyan]")
            url = "https://min-api.cryptocompare.com/data/v2/news/"
            headers = {"authorization": f"Apikey {self.config.cryptocompare_api_key}"}
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    news_count = len(data.get("Data", []))
                    console.print(f"[green]Successfully fetched {news_count} articles from CryptoCompare[/green]")
                    return data.get("Data", [])
                else:
                    console.print(f"[red]CryptoCompare API error: {response.status}[/red]")
                    return []
        except Exception as e:
            console.print(f"[red]Error fetching CryptoCompare news: {str(e)}[/red]")
            return []

    async def fetch_cryptopanic_news(self) -> List[Dict]:
        """Fetch news from CryptoPanic"""
        if not self.config.cryptopanic_api_key:
            console.print("[yellow]CryptoPanic API key not configured[/yellow]")
            return []

        try:
            console.print("[cyan]Fetching news from CryptoPanic...[/cyan]")
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.config.cryptopanic_api_key}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    news_count = len(data.get("results", []))
                    console.print(f"[green]Successfully fetched {news_count} articles from CryptoPanic[/green]")
                    return data.get("results", [])
                else:
                    console.print(f"[red]CryptoPanic API error: {response.status}[/red]")
                    return []
        except Exception as e:
            console.print(f"[red]Error fetching CryptoPanic news: {str(e)}[/red]")
            return []

    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to 1 scale)"""
        if not hasattr(self, '_sentiment_analyzer'):
            self._sentiment_analyzer = SentimentAnalyzer()
        
        try:
            return self._sentiment_analyzer.analyze(text)
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Fallback to basic sentiment analysis if something goes wrong
            positive_words = {"bullish", "surge", "gain", "positive", "up", "high", "rise"}
            negative_words = {"bearish", "crash", "drop", "negative", "down", "low", "fall"}
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            if positive_count == 0 and negative_count == 0:
                return 0
            
            return (positive_count - negative_count) / (positive_count + negative_count)

    def calculate_news_impact(self, news_items: List[Dict]) -> float:
        """Calculate overall news impact"""
        if not news_items:
            return 0.0

        sentiments = []
        weights = []

        for item in news_items:
            # Extract text and calculate base sentiment
            text = f"{item.get('title', '')} {item.get('body', '')}"
            sentiment = self.analyze_sentiment(text)
            
            # Calculate weight based on time
            age = time.time() - item.get('published_on', time.time())
            time_weight = max(0, 1 - (age / self.config.max_news_age))
            
            # Adjust weight based on source reliability
            source_weight = item.get('source_reliability', 1.0)
            
            final_weight = time_weight * source_weight
            
            sentiments.append(sentiment)
            weights.append(final_weight)

        # Calculate weighted average
        return np.average(sentiments, weights=weights)

    async def update_sentiment(self) -> Dict:
        """Update current sentiment analysis"""
        if not self.config.enabled:
            console.print("[yellow]News sentiment analysis is currently disabled[/yellow]")
            return {
                "sentiment": 0.0,
                "emergency": False,
                "last_update": self.last_update.isoformat(),
                "status": "disabled"
            }

        try:
            console.print("\n[bold cyan]Updating News Sentiment Analysis[/bold cyan]")
            
            # Fetch news from multiple sources
            cryptocompare_news = await self.fetch_cryptocompare_news()
            cryptopanic_news = await self.fetch_cryptopanic_news()
            
            # Combine news from all sources
            all_news = cryptocompare_news + cryptopanic_news
            
            # Filter by keywords
            filtered_news = [
                news for news in all_news
                if any(keyword in news.get('title', '').lower() for keyword in self.config.keywords)
            ]
            
            console.print(f"[cyan]Processing {len(filtered_news)} relevant news articles...[/cyan]")
            
            # Calculate sentiment
            self.current_sentiment = self.calculate_news_impact(filtered_news)
            
            # Check for emergency signals
            self.emergency_signal = self.current_sentiment <= self.config.emergency_threshold
            
            self.last_update = datetime.now()
            
            # Create sentiment status panel
            sentiment_color = "green" if self.current_sentiment > 0 else "red" if self.current_sentiment < 0 else "yellow"
            sentiment_status = Panel(
                f"Sentiment Score: [{sentiment_color}]{self.current_sentiment:+.4f}[/]\n"
                f"Emergency Signal: {'[red]YES[/]' if self.emergency_signal else '[green]NO[/]'}\n"
                f"Articles Analyzed: {len(filtered_news)}",
                title="Sentiment Analysis Results",
                border_style="cyan"
            )
            console.print(sentiment_status)
            
            return {
                "sentiment": self.current_sentiment,
                "emergency": self.emergency_signal,
                "last_update": self.last_update.isoformat(),
                "status": "updated",
                "news_count": len(filtered_news)
            }
            
        except Exception as e:
            error_msg = f"Error updating sentiment: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            logger.error(error_msg)
            return {
                "sentiment": self.current_sentiment,
                "emergency": self.emergency_signal,
                "last_update": self.last_update.isoformat(),
                "status": "error",
                "error": str(e)
            }

    def should_update(self) -> bool:
        """Check if sentiment should be updated"""
        if not self.config.enabled:
            return False
        
        time_since_update = (datetime.now() - self.last_update).total_seconds()
        return time_since_update >= self.config.update_interval

    def adjust_signal(self, technical_signal: float) -> float:
        """Adjust technical signal based on sentiment"""
        if not self.config.enabled:
            return technical_signal
            
        if self.emergency_signal:
            return -1.0  # Force sell signal on emergency
            
        # Blend technical and sentiment signals
        sentiment_weight = min(abs(self.current_sentiment), 0.3)  # Cap sentiment influence
        technical_weight = 1 - sentiment_weight
        
        return (technical_signal * technical_weight) + (self.current_sentiment * sentiment_weight)

    def save_state(self, filepath: str = "news_analyzer_state.json"):
        """Save analyzer state"""
        try:
            state = {
                "last_update": self.last_update.isoformat(),
                "current_sentiment": self.current_sentiment,
                "emergency_signal": self.emergency_signal,
                "config": {
                    "enabled": self.config.enabled,
                    "sentiment_threshold": self.config.sentiment_threshold,
                    "emergency_threshold": self.config.emergency_threshold,
                    "keywords": self.config.keywords
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=4)
            console.print(f"[green]Successfully saved analyzer state to {filepath}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving analyzer state: {str(e)}[/red]")
            logger.error(f"Error saving analyzer state: {str(e)}")

    def load_state(self, filepath: str = "news_analyzer_state.json"):
        """Load analyzer state"""
        try:
            if Path(filepath).exists():
                console.print(f"[cyan]Loading analyzer state from {filepath}...[/cyan]")
                with open(filepath, 'r') as f:
                    state = json.load(f)
                
                self.last_update = datetime.fromisoformat(state['last_update'])
                self.current_sentiment = state['current_sentiment']
                self.emergency_signal = state['emergency_signal']
                
                # Update config
                config_state = state.get('config', {})
                self.config.enabled = config_state.get('enabled', self.config.enabled)
                self.config.sentiment_threshold = config_state.get('sentiment_threshold', self.config.sentiment_threshold)
                self.config.emergency_threshold = config_state.get('emergency_threshold', self.config.emergency_threshold)
                self.config.keywords = config_state.get('keywords', self.config.keywords)
                
                console.print("[green]Successfully loaded analyzer state[/green]")
                
                # Show current state
                state_panel = Panel(
                    f"Last Update: {self.last_update.isoformat()}\n"
                    f"Current Sentiment: {self.current_sentiment:+.4f}\n"
                    f"Emergency Signal: {'[red]YES[/]' if self.emergency_signal else '[green]NO[/]'}",
                    title="Current Analyzer State",
                    border_style="cyan"
                )
                console.print(state_panel)
        except Exception as e:
            error_msg = f"Error loading analyzer state: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            logger.error(error_msg)

    def toggle(self, enabled: bool = None):
        """Toggle news analysis on/off"""
        if enabled is None:
            self.config.enabled = not self.config.enabled
        else:
            self.config.enabled = enabled
        
        status = "[green]enabled[/green]" if self.config.enabled else "[yellow]disabled[/yellow]"
        console.print(f"News sentiment analysis {status}")
        
        # Save state and show current configuration
        self.save_state()
        config_panel = Panel(
            f"Enabled: {status}\n"
            f"Sentiment Threshold: {self.config.sentiment_threshold}\n"
            f"Emergency Threshold: {self.config.emergency_threshold}\n"
            f"Keywords: {', '.join(self.config.keywords)}",
            title="News Analyzer Configuration",
            border_style="blue"
        )
        console.print(config_panel)