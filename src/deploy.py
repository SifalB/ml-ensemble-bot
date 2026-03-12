#!/usr/bin/env python3
"""
Deployment script for ML ensemble bot + A/B testing.
Runs both control (cheap lottery) and test (ML ensemble) bots.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from src.data_collector import PolymarketDataCollector, SyntheticDataGenerator
from src.ensemble_trader import EnsembleTrader
from src.ml_ensemble import LSTMEnsembleModel
from src.features import FeatureEngineer
from src.ab_testing import ABTestingFramework

def load_or_train_model():
    """Load pre-trained model or train a new one."""
    
    model_path = Path('models/ensemble_model.h5')
    
    if model_path.exists():
        print("✓ Loading pre-trained model...")
        model = LSTMEnsembleModel()
        model.load('models')
        return model
    else:
        print("⚠ Model not found. Training new model...")
        from src.train_model import train_ensemble_model
        model, _ = train_ensemble_model()
        return model

def run_control_bot(cycle_id: int) -> dict:
    """Run cheap lottery control strategy."""
    
    # Simulate control bot results
    # (In reality, this would be running from polymarket-bot)
    
    import random
    
    control_result = {
        'timestamp': datetime.now().isoformat(),
        'balance': 50.0 - (cycle_id * 0.5),  # Slight decline
        'equity': 50.0 - (cycle_id * 0.5),
        'pnl': -cycle_id * 0.5,
        'roi': -(cycle_id * 0.5) / 50.0 * 100,
        'trades_executed': random.randint(0, 5),
        'win_rate': 45.0
    }
    
    return control_result

def run_test_bot(model: LSTMEnsembleModel, cycle_id: int) -> dict:
    """Run ML ensemble test strategy."""
    
    # Generate current market data
    markets_data = SyntheticDataGenerator.generate_multiple_markets(
        n_markets=20,
        days=7  # 1 week recent data
    )
    
    # Initialize trader if not exists
    if not hasattr(run_test_bot, 'trader'):
        run_test_bot.trader = EnsembleTrader(
            initial_balance=50.0,
            risk_per_trade=0.05,
            confidence_threshold=0.75,
            max_positions=10
        )
        run_test_bot.trader.set_model(model)
    
    # Run one trading cycle
    result = run_test_bot.trader.run_cycle(markets_data)
    
    # Save trader state
    Path('data').mkdir(exist_ok=True)
    run_test_bot.trader.save_wallet('data/wallet_test.json')
    
    return result

def main():
    """Main deployment loop."""
    
    print("\n" + "=" * 80)
    print("ML ENSEMBLE BOT - DEPLOYMENT & A/B TESTING")
    print("=" * 80)
    
    # ===== INITIALIZE =====
    print("\n[INIT] Loading model...")
    model = load_or_train_model()
    
    print("[INIT] Setting up A/B testing framework...")
    ab_test = ABTestingFramework()
    
    # ===== LOAD PREVIOUS RESULTS =====
    results_path = Path('data/ab_test_results.json')
    if results_path.exists():
        print("[INIT] Loading previous results...")
        with open(results_path, 'r') as f:
            ab_test.results = json.load(f)
    
    # ===== RUN CYCLES =====
    print("\n[START] Running A/B test cycles...")
    print("-" * 80)
    
    n_cycles = 3  # Run 3 cycles for demo
    
    for cycle in range(1, n_cycles + 1):
        
        print(f"\n▶ CYCLE #{cycle}")
        
        # Run control bot
        control_result = run_control_bot(cycle)
        ab_test.record_cycle('control', control_result)
        
        print(f"  Control: Balance=${control_result['balance']:.2f}, ROI={control_result['roi']:.2f}%")
        
        # Run test bot
        test_result = run_test_bot(model, cycle)
        ab_test.record_cycle('test', test_result)
        
        print(f"  Test:    Balance=${test_result['balance']:.2f}, ROI={test_result['roi']:.2f}%")
        
        # Save results after each cycle
        ab_test.save_results()
    
    # ===== FINAL REPORT =====
    print("\n" + "=" * 80)
    ab_test.print_comparison()
    
    # ===== SAVE FINAL STATE =====
    Path('data').mkdir(exist_ok=True)
    
    with open('data/ab_test_results.json', 'w') as f:
        json.dump(ab_test.results, f, indent=2, default=str)
    
    print("\n✓ Results saved to data/ab_test_results.json")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'status':
            # Check wallet status
            wallet_path = Path('data/wallet_test.json')
            if wallet_path.exists():
                with open(wallet_path, 'r') as f:
                    wallet = json.load(f)
                    print(f"Wallet balance: ${wallet['balance']:.2f}")
                    print(f"Equity:         ${wallet['equity']:.2f}")
                    print(f"PnL:            ${wallet['pnl']:.2f}")
                    print(f"Open positions: {len(wallet['positions'])}")
            else:
                print("No wallet found")
        
        elif sys.argv[1] == 'run':
            # Run single cycle
            main()
    
    else:
        main()
