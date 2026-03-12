"""
Data collection for ML ensemble bot.
Fetches historical price data from Polymarket and generates synthetic training data.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json

class PolymarketDataCollector:
    """Fetch and process Polymarket data."""
    
    def __init__(self, api_base: str = "https://gamma-api.polymarket.com"):
        self.api_base = api_base
        self.session = requests.Session()
    
    def get_markets(self, limit: int = 100, closed: bool = False) -> List[Dict]:
        """Fetch list of active markets."""
        
        params = {
            'limit': limit,
            'closed': closed,
            'orderBy': 'volume24hUsd',
            'order': 'desc'
        }
        
        try:
            resp = self.session.get(f"{self.api_base}/markets", params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []
    
    def get_order_book(self, market_id: str) -> Dict:
        """Fetch current order book for a market."""
        
        try:
            resp = self.session.get(
                f"{self.api_base}/order-book",
                params={'market_id': market_id},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching order book {market_id}: {e}")
            return {}
    
    def get_market_history(self, market_id: str, limit: int = 300) -> List[Dict]:
        """Fetch price history for a market."""
        
        try:
            resp = self.session.get(
                f"{self.api_base}/markets/{market_id}",
                params={'limit': limit},
                timeout=10
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"Error fetching history {market_id}: {e}")
            return []
    
    def to_ohlcv_dataframe(self, market_data: List[Dict], market_id: str) -> pd.DataFrame:
        """Convert Polymarket data to OHLCV format."""
        
        if not market_data:
            return pd.DataFrame()
        
        # Polymarket returns quote prices for YES/NO outcomes
        # We'll use YES price as our price signal
        
        df = pd.DataFrame(market_data)
        
        if 'yes_price' not in df.columns or 'volume' not in df.columns:
            return pd.DataFrame()
        
        # Create OHLCV structure
        df['timestamp'] = pd.to_datetime(df.get('timestamp', datetime.now()), utc=True)
        df['close'] = df['yes_price']
        df['open'] = df.get('open_price', df['close'])
        df['high'] = df.get('high_price', df['close'].rolling(5).max())
        df['low'] = df.get('low_price', df['close'].rolling(5).min())
        df['volume'] = df.get('volume', 0)
        df['market_id'] = market_id
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'market_id']].sort_values('timestamp')


class SyntheticDataGenerator:
    """Generate synthetic market data for training (realistic patterns)."""
    
    @staticmethod
    def generate_geometric_brownian_motion(
        S0: float = 0.5,
        mu: float = 0.0002,
        sigma: float = 0.15,
        days: int = 365,
        intervals_per_day: int = 24
    ) -> pd.DataFrame:
        """
        Generate synthetic price data using Geometric Brownian Motion.
        Models realistic price movement with drift and volatility.
        
        GBM: dS = mu*S*dt + sigma*S*dW
        """
        
        dt = 1 / (intervals_per_day * 365)  # Time step
        N = days * intervals_per_day
        
        # Generate random returns
        dW = np.random.normal(0, np.sqrt(dt), N)
        
        # GBM simulation
        S = np.zeros(N)
        S[0] = S0
        
        for t in range(1, N):
            S[t] = S[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[t])
        
        # Create OHLCV
        timestamps = pd.date_range(start='2020-01-01', periods=N, freq='h')
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'close': S,
            'open': S * (1 + np.random.normal(0, 0.01, N)),
            'high': S * (1 + np.abs(np.random.normal(0, 0.02, N))),
            'low': S * (1 - np.abs(np.random.normal(0, 0.02, N))),
            'volume': np.random.exponential(1000, N)
        })
        
        # Ensure OHLC relationships
        df['high'] = df[['open', 'high', 'close']].max(axis=1)
        df['low'] = df[['open', 'low', 'close']].min(axis=1)
        
        return df
    
    @staticmethod
    def generate_multiple_markets(
        n_markets: int = 30,
        days: int = 300
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for multiple markets."""
        
        markets = {}
        
        for i in range(n_markets):
            # Vary parameters slightly per market
            mu = np.random.uniform(-0.0005, 0.0005)
            sigma = np.random.uniform(0.10, 0.25)
            S0 = np.random.uniform(0.2, 0.8)
            
            df = SyntheticDataGenerator.generate_geometric_brownian_motion(
                S0=S0,
                mu=mu,
                sigma=sigma,
                days=days,
                intervals_per_day=4  # 4 data points per day (every 6 hours)
            )
            
            markets[f"market_{i}"] = df
        
        return markets


def prepare_training_data(
    df: pd.DataFrame,
    test_split: float = 0.2
) -> tuple:
    """
    Split data into train/val/test sets with no data leakage.
    
    Returns:
        (train_df, val_df, test_df)
    """
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    n = len(df)
    
    # 80% train+val, 20% test
    split_val = int(n * (1 - test_split))
    
    train_val = df.iloc[:split_val].copy()
    test = df.iloc[split_val:].copy()
    
    # Split train_val into 80/20
    split_train = int(len(train_val) * 0.8)
    train = train_val.iloc[:split_train].copy()
    val = train_val.iloc[split_train:].copy()
    
    return train, val, test
