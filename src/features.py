"""
Feature engineering for ML ensemble bot.
Calculates 38+ technical indicators for LSTM model.
"""

import pandas as pd
import numpy as np
from typing import Tuple
import ta  # Technical Analysis library

class FeatureEngineer:
    """Calculate technical indicators for market data."""
    
    def __init__(self, lookback_window: int = 60):
        self.lookback = lookback_window
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 38 technical indicators to dataframe.
        
        Input: DataFrame with OHLCV (Open, High, Low, Close, Volume)
        Output: DataFrame with price + 38 indicators + Target
        """
        
        # Make a copy to avoid modifying original
        data = df.copy()
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # ===== TARGET =====
        # Binary: does price go UP tomorrow?
        data['Target_1D'] = (data['close'].shift(-1) > data['close']).astype(int)
        
        # ===== MOVING AVERAGES =====
        data['MA_10'] = data['close'].rolling(10).mean()
        data['MA_20'] = data['close'].rolling(20).mean()
        data['MA_30'] = data['close'].rolling(30).mean()
        data['MA_50'] = data['close'].rolling(50).mean()
        
        # ===== MOMENTUM =====
        data['Momentum_5'] = data['close'] - data['close'].shift(5)
        data['Momentum_20'] = data['close'] - data['close'].shift(20)
        data['Momentum_Ratio'] = data['Momentum_5'] / (data['Momentum_20'] + 1e-8)
        
        # ===== RSI (Relative Strength Index) =====
        data['RSI'] = ta.momentum.rsi(data['close'], window=14)
        
        # ===== MACD (Moving Average Convergence Divergence) =====
        macd = ta.trend.MACD(data['close'])
        data['MACD'] = macd.macd()
        data['MACD_Signal'] = macd.macd_signal()
        data['MACD_Diff'] = macd.macd_diff()
        
        # ===== BOLLINGER BANDS =====
        bb = ta.volatility.BollingerBands(data['close'])
        data['BB_Upper'] = bb.bollinger_hband()
        data['BB_Lower'] = bb.bollinger_lband()
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['close'] - data['BB_Lower']) / (data['BB_Width'] + 1e-8)
        
        # ===== VOLATILITY =====
        data['Volatility_10'] = data['close'].pct_change().rolling(10).std()
        data['Volatility_20'] = data['close'].pct_change().rolling(20).std()
        data['Volatility_30'] = data['close'].pct_change().rolling(30).std()
        
        # ===== VOLUME-BASED =====
        data['Volume_MA_10'] = data['volume'].rolling(10).mean()
        data['Volume_Ratio'] = data['volume'] / (data['Volume_MA_10'] + 1e-8)
        
        # ===== OBV (On-Balance Volume) =====
        data['OBV'] = self._calculate_obv(data)
        data['OBV_MA'] = data['OBV'].rolling(10).mean()
        
        # ===== PRICE CHANGES =====
        data['Returns_1'] = data['close'].pct_change(1)
        data['Returns_5'] = data['close'].pct_change(5)
        data['Returns_20'] = data['close'].pct_change(20)
        
        # ===== VOLATILITY OF RETURNS =====
        data['Return_Std_10'] = data['Returns_1'].rolling(10).std()
        data['Return_Std_20'] = data['Returns_1'].rolling(20).std()
        
        # ===== TREND STRENGTH (ADX) =====
        adx = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], window=14)
        data['ADX'] = adx.adx()
        
        # ===== ATR (Average True Range) - Volatility indicator =====
        atr = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'])
        data['ATR'] = atr.average_true_range()
        
        # ===== STOCHASTIC OSCILLATOR =====
        stoch = ta.momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['Stoch_K'] = stoch.stoch()
        data['Stoch_D'] = stoch.stoch_signal()
        
        # ===== CCI (Commodity Channel Index) =====
        data['CCI'] = ta.momentum.cci(data['high'], data['low'], data['close'], window=20)
        
        # ===== Normalized price position =====
        data['Close_Min_20'] = data['close'].rolling(20).min()
        data['Close_Max_20'] = data['close'].rolling(20).max()
        data['Price_Position'] = (data['close'] - data['Close_Min_20']) / (data['Close_Max_20'] - data['Close_Min_20'] + 1e-8)
        
        # Fill NaN values
        data = data.fillna(method='bfill').fillna(method='ffill')
        
        return data
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume manually."""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def get_feature_names(self) -> list:
        """Return list of all feature names."""
        return [
            'MA_10', 'MA_20', 'MA_30', 'MA_50',
            'Momentum_5', 'Momentum_20', 'Momentum_Ratio',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Diff',
            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
            'Volatility_10', 'Volatility_20', 'Volatility_30',
            'Volume_MA_10', 'Volume_Ratio', 'OBV', 'OBV_MA',
            'Returns_1', 'Returns_5', 'Returns_20',
            'Return_Std_10', 'Return_Std_20',
            'ADX', 'ATR', 'Stoch_K', 'Stoch_D', 'CCI',
            'Close_Min_20', 'Close_Max_20', 'Price_Position'
        ]
    
    def normalize_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        """
        Normalize train and test data using train statistics.
        Prevents data leakage.
        """
        from sklearn.preprocessing import StandardScaler
        
        feature_names = self.get_feature_names()
        
        # Fit scaler on training data only
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_names])
        
        # Apply same scaler to test data
        test_scaled = scaler.transform(test_df[feature_names])
        
        return train_scaled, test_scaled, scaler
