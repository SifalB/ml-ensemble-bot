"""
Ensemble Trading Bot - integrates ML predictions with Polymarket API.
Paper trading with risk management.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from src.ml_ensemble import LSTMEnsembleModel
from src.features import FeatureEngineer
from src.data_collector import PolymarketDataCollector

class EnsembleTrader:
    """ML ensemble-based trading bot."""
    
    def __init__(self, 
                 initial_balance: float = 50.0,
                 risk_per_trade: float = 0.05,
                 confidence_threshold: float = 0.75,
                 max_positions: int = 10):
        """
        Initialize ensemble trader.
        
        Args:
            initial_balance: Starting wallet in USD
            risk_per_trade: Max % of wallet to risk per trade (0.05 = 5%)
            confidence_threshold: Min confidence for BUY signal (0.75 = 75%)
            max_positions: Max open positions at once
        """
        
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.confidence_threshold = confidence_threshold
        self.max_positions = max_positions
        
        self.positions = []  # List of open trades
        self.trades_history = []  # All executed trades
        self.pnl = 0.0
        self.equity = initial_balance
        
        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.data_collector = PolymarketDataCollector()
    
    def set_model(self, model: LSTMEnsembleModel):
        """Attach trained ML model."""
        self.model = model
    
    def run_cycle(self, markets_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Execute one trading cycle.
        
        1. Fetch latest market data
        2. Generate features for all markets
        3. Run ML predictions
        4. Filter by confidence threshold
        5. Execute top N trades with risk management
        6. Update portfolio
        """
        
        if self.model is None:
            return {'error': 'Model not set'}
        
        cycle_start = datetime.now()
        signals = {}
        
        # ===== STEP 1: Generate features for all markets =====
        for market_id, df in markets_data.items():
            
            # Add features
            df_with_features = self.feature_engineer.add_features(df)
            
            if len(df_with_features) < 30:
                continue  # Not enough data
            
            # ===== STEP 2: Prepare sequences for model =====
            feature_names = self.feature_engineer.get_feature_names()
            X = df_with_features[feature_names].values[-30:].reshape(1, 30, -1)
            
            # ===== STEP 3: Get ML prediction with uncertainty =====
            mean, std, _ = self.model.mc_predict(X)
            
            confidence = float(mean[0])
            uncertainty = float(std[0])
            
            # ===== STEP 4: Filter by confidence =====
            if confidence > self.confidence_threshold and uncertainty < 0.15:
                signal = 'BUY'
            else:
                signal = 'HOLD'
            
            signals[market_id] = {
                'signal': signal,
                'confidence': confidence,
                'uncertainty': uncertainty,
                'price': float(df_with_features['close'].iloc[-1])
            }
        
        # ===== STEP 5: Sort by confidence and execute top positions =====
        buy_signals = [
            (m_id, data) for m_id, data in signals.items()
            if data['signal'] == 'BUY'
        ]
        buy_signals.sort(key=lambda x: x[1]['confidence'], reverse=True)
        
        # ===== STEP 6: Execute trades =====
        trades_executed = 0
        
        for market_id, signal_data in buy_signals:
            
            # Check position limits
            if len(self.positions) >= self.max_positions:
                break
            
            if self.balance <= 0:
                break
            
            # Calculate position size based on risk
            position_size = self.balance * self.risk_per_trade
            position_size = min(position_size, 5.0)  # Max $5 per position
            
            # Execute trade
            trade = {
                'timestamp': cycle_start.isoformat(),
                'market_id': market_id,
                'action': 'BUY',
                'entry_price': signal_data['price'],
                'size': position_size,
                'confidence': signal_data['confidence'],
                'uncertainty': signal_data['uncertainty'],
                'status': 'OPEN'
            }
            
            self.positions.append(trade)
            self.balance -= position_size
            trades_executed += 1
        
        # ===== STEP 7: Resolve expired positions (simulation) =====
        # In reality, positions resolve at market outcome
        # For now, we'll simulate 1-day holding period
        
        closed_trades = []
        for pos in self.positions[:]:
            
            entry_time = pd.to_datetime(pos['timestamp'])
            days_held = (cycle_start - entry_time).days
            
            if days_held >= 1:  # Hold for 1 day
                
                # Simulate outcome: 60% win rate based on confidence
                win_probability = pos['confidence']
                outcome = np.random.random() < win_probability
                
                if outcome:
                    # Win: position becomes $1.00 (Polymarket resolution)
                    pnl = (1.0 - pos['entry_price']) * pos['size']
                else:
                    # Loss: position becomes $0.00
                    pnl = -pos['entry_price'] * pos['size']
                
                closed_trade = pos.copy()
                closed_trade['status'] = 'CLOSED'
                closed_trade['exit_price'] = 1.0 if outcome else 0.0
                closed_trade['pnl'] = pnl
                closed_trade['outcome'] = 'WIN' if outcome else 'LOSS'
                closed_trade['close_timestamp'] = cycle_start.isoformat()
                
                self.trades_history.append(closed_trade)
                self.pnl += pnl
                self.balance += (pos['size'] + pnl)
                self.positions.remove(pos)
                
                closed_trades.append(closed_trade)
        
        self.equity = self.balance + sum(p['size'] for p in self.positions)
        
        # ===== RETURN CYCLE METRICS =====
        
        return {
            'timestamp': cycle_start.isoformat(),
            'balance': round(self.balance, 2),
            'equity': round(self.equity, 2),
            'pnl': round(self.pnl, 2),
            'roi': round((self.equity - self.initial_balance) / self.initial_balance * 100, 2),
            'open_positions': len(self.positions),
            'trades_executed': trades_executed,
            'closed_trades': len(closed_trades),
            'signals_total': len(signals),
            'buy_signals': len([s for s in signals.values() if s['signal'] == 'BUY'])
        }
    
    def get_stats(self) -> Dict:
        """Calculate trader statistics."""
        
        if not self.trades_history:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        trades_df = pd.DataFrame(self.trades_history)
        
        wins = trades_df[trades_df['outcome'] == 'WIN']
        losses = trades_df[trades_df['outcome'] == 'LOSS']
        
        total_wins = len(wins)
        total_losses = len(losses)
        total_trades = len(trades_df)
        
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
        
        total_profit = wins['pnl'].sum()
        total_loss = losses['pnl'].sum()
        
        profit_factor = total_profit / abs(total_loss) if total_loss != 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': total_wins,
            'losing_trades': total_losses,
            'win_rate': round(total_wins / total_trades * 100, 2) if total_trades > 0 else 0,
            'avg_win': round(avg_win, 4),
            'avg_loss': round(avg_loss, 4),
            'profit_factor': round(profit_factor, 2),
            'total_pnl': round(self.pnl, 2),
            'equity': round(self.equity, 2)
        }
    
    def save_wallet(self, path: str):
        """Save trading state to JSON."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'balance': self.balance,
            'equity': self.equity,
            'pnl': self.pnl,
            'positions': self.positions,
            'trades_history': self.trades_history
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_wallet(self, path: str):
        """Load trading state from JSON."""
        try:
            with open(path, 'r') as f:
                state = json.load(f)
                self.balance = state.get('balance', self.initial_balance)
                self.equity = state.get('equity', self.initial_balance)
                self.pnl = state.get('pnl', 0.0)
                self.positions = state.get('positions', [])
                self.trades_history = state.get('trades_history', [])
        except:
            pass
