"""
A/B Testing Framework - Compare ML ensemble vs control bot.
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path

class ABTestingFramework:
    """Track and compare performance of two strategies."""
    
    def __init__(self):
        self.results = {
            'control': [],  # Cheap lottery results
            'test': []      # ML ensemble results
        }
    
    def record_cycle(self, group: str, cycle_data: dict):
        """Record one trading cycle result."""
        
        if group not in self.results:
            return
        
        self.results[group].append({
            'timestamp': cycle_data.get('timestamp', datetime.now().isoformat()),
            'balance': cycle_data.get('balance'),
            'equity': cycle_data.get('equity'),
            'pnl': cycle_data.get('pnl'),
            'roi': cycle_data.get('roi'),
            'trades_executed': cycle_data.get('trades_executed'),
            'win_rate': cycle_data.get('win_rate')
        })
    
    def get_comparison_stats(self) -> dict:
        """Compare performance of both groups."""
        
        if not self.results['control'] or not self.results['test']:
            return {'error': 'Insufficient data for comparison'}
        
        control_df = pd.DataFrame(self.results['control'])
        test_df = pd.DataFrame(self.results['test'])
        
        comparison = {
            'control': {
                'total_trades': control_df['trades_executed'].sum(),
                'avg_roi': control_df['roi'].mean(),
                'final_equity': control_df['equity'].iloc[-1] if len(control_df) > 0 else 0,
                'max_drawdown': (control_df['equity'].min() - control_df['equity'].iloc[0]) / control_df['equity'].iloc[0] * 100,
                'sharpe_ratio': self._calculate_sharpe(control_df['roi']),
                'cycles': len(control_df)
            },
            'test': {
                'total_trades': test_df['trades_executed'].sum(),
                'avg_roi': test_df['roi'].mean(),
                'final_equity': test_df['equity'].iloc[-1] if len(test_df) > 0 else 0,
                'max_drawdown': (test_df['equity'].min() - test_df['equity'].iloc[0]) / test_df['equity'].iloc[0] * 100,
                'sharpe_ratio': self._calculate_sharpe(test_df['roi']),
                'cycles': len(test_df)
            }
        }
        
        # Winner
        if comparison['test']['avg_roi'] > comparison['control']['avg_roi']:
            winner = 'test'
            edge = comparison['test']['avg_roi'] - comparison['control']['avg_roi']
        else:
            winner = 'control'
            edge = comparison['control']['avg_roi'] - comparison['test']['avg_roi']
        
        comparison['winner'] = winner
        comparison['edge_bps'] = round(edge * 100, 2)  # Basis points
        
        return comparison
    
    def _calculate_sharpe(self, returns: pd.Series, rf_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0
        
        excess_returns = returns - rf_rate / 365  # Daily risk-free rate
        sharpe = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0
        return sharpe
    
    def save_results(self, path: str = 'data/ab_test_results.json'):
        """Save A/B test results to file."""
        
        Path('data').mkdir(exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
    
    def print_comparison(self):
        """Print comparison report."""
        
        stats = self.get_comparison_stats()
        
        if 'error' in stats:
            print(stats['error'])
            return
        
        print("\n" + "=" * 80)
        print("A/B TEST COMPARISON REPORT")
        print("=" * 80)
        
        print("\n📊 CONTROL GROUP (Cheap Lottery):")
        print("-" * 40)
        print(f"  Total Trades:    {stats['control']['total_trades']}")
        print(f"  Avg ROI:         {stats['control']['avg_roi']:.2f}%")
        print(f"  Final Equity:    ${stats['control']['final_equity']:.2f}")
        print(f"  Max Drawdown:    {stats['control']['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:    {stats['control']['sharpe_ratio']:.3f}")
        print(f"  Cycles:          {stats['control']['cycles']}")
        
        print("\n🧠 TEST GROUP (ML Ensemble):")
        print("-" * 40)
        print(f"  Total Trades:    {stats['test']['total_trades']}")
        print(f"  Avg ROI:         {stats['test']['avg_roi']:.2f}%")
        print(f"  Final Equity:    ${stats['test']['final_equity']:.2f}")
        print(f"  Max Drawdown:    {stats['test']['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio:    {stats['test']['sharpe_ratio']:.3f}")
        print(f"  Cycles:          {stats['test']['cycles']}")
        
        print("\n🏆 VERDICT:")
        print("-" * 40)
        winner_name = "ML Ensemble" if stats['winner'] == 'test' else "Cheap Lottery"
        print(f"  Winner:          {winner_name}")
        print(f"  Edge:            {stats['edge_bps']} bps ({stats['edge_bps']/100:.2f}%)")
        
        print("\n" + "=" * 80)
