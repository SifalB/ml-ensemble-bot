# Live Deployment - ML Ensemble Bot with Real Polymarket Data

**Status:** ✅ Phase 2 Live Integration Ready  
**Date:** 2026-03-12  
**Repository:** https://github.com/SifalB/ml-ensemble-bot

## Overview

The ML ensemble bot is now connected to **real Polymarket API** and ready for live trading with real market data.

### What Changed (Phase 2)

#### New Modules
- **`src/live_integration.py`** - Live Polymarket API client
  - `LivePolymarketIntegration` - Connect to real Gamma + CLOB APIs
  - `LiveDataStreamer` - Stream and cache market data
  - Supports market scanning, price fetching, order book reading

- **`src/deploy_live.py`** - Live deployment script
  - Fetch real Polymarket data every cycle
  - Run ML ensemble predictions on live markets
  - A/B test ML vs control bot on real data
  - Generate hourly reports

#### API Connection Details
```python
# Connect to real Polymarket
api = LivePolymarketIntegration()

# Scan 100+ active markets
markets = api.get_active_markets(limit=100)

# Get current prices + spreads
prices = api.get_market_prices(market_id)
```

#### Live Data Features
- ✅ Fetch 100+ active markets per cycle
- ✅ Real bid/ask prices from order book
- ✅ Live spread calculation (arbitrage detection)
- ✅ Volume tracking (24h)
- ✅ Error handling + caching

---

## Quick Start

### 1. Train Model (if not already done)
```bash
python -m src.train_model
```
Takes ~5 minutes. Produces trained LSTM at `models/ensemble_model.h5`

### 2. Test Live API Connection
```bash
python -m src.deploy_live test
```
Expected output:
```
✓ Connected! Found 100+ active markets
✓ Market prices: YES bid/ask, spread calculated
✓ LIVE API CONNECTION WORKING
```

### 3. Start Live Deployment
```bash
python -m src.deploy_live
```
Runs continuous hourly cycles:
- Fetch live market data
- Generate ML predictions
- Execute trades
- Record A/B test results
- Report metrics

Press `Ctrl+C` to stop.

### 4. Monitor Wallet
```bash
python -m src.deploy_live status
```
Shows:
- Current balance
- Total equity
- Cumulative P&L
- Open positions

---

## How It Works

### Live Cycle Flow

```
Polymarket API
     ↓
Fetch 100+ active markets
     ↓
Extract bid/ask prices
     ↓
Calculate spreads
     ↓
Feature engineering (38 indicators)
     ↓
MC Dropout inference (50 passes)
     ↓
Confidence filtering (≥75%)
     ↓
Trade execution (risk management)
     ↓
A/B test comparison
     ↓
Record results + metrics
     ↓
Report to user
```

### Expected Cycle Time
- Fetch data: ~5 seconds
- ML inference: ~3 seconds  
- Trade execution: ~2 seconds
- **Total: ~10 seconds per cycle**

---

## Live Performance Metrics

### Output Per Cycle
```json
{
  "timestamp": "2026-03-12T09:15:00Z",
  "markets_fetched": 100,
  "markets_predicted": 87,
  "buy_signals": 3,
  "hold_signals": 84,
  "trades_executed": 2,
  "balance": 49.95,
  "equity": 49.95,
  "pnl": -0.05,
  "roi": -0.10,
  "stats": {
    "total_trades": 2,
    "win_rate": 50.0,
    "sharpe_ratio": 0.5,
    "max_drawdown": -0.10
  }
}
```

---

## A/B Test Results

After live trading, compare:

**Control Bot (Cheap Lottery)**
- Random tail event buys
- Baseline strategy

**Test Bot (ML Ensemble)**
- ML-predicted high-confidence signals
- Expected outperformance

### Success Metrics
- ✅ Test ROI > Control ROI
- ✅ Test Sharpe > 1.5
- ✅ Test win rate > 60%
- ✅ Edge > 100 basis points (1%)

### Reports
Final A/B test report saved to `data/ab_test_results.json`:
```bash
python -c "
import json
with open('data/ab_test_results.json') as f:
    results = json.load(f)
    print(f\"Control ROI: {results['control'][-1]['roi']}%\")
    print(f\"Test ROI: {results['test'][-1]['roi']}%\")
"
```

---

## Risk Management (Live)

Same rules as paper trading:

| Setting | Value | Reason |
|---------|-------|--------|
| Initial balance | $50 | Safe test amount |
| Risk per trade | 5% | $2.50 max per position |
| Max positions | 10 | Diversification |
| Global stop loss | -30% | $35 minimum |
| Confidence threshold | 75% | High-conviction trades only |
| Uncertainty filter | std < 0.15 | Reject uncertain predictions |

---

## Live Data Sources

### Polymarket API Endpoints

1. **Gamma API** (Market data)
   - Base: `https://gamma-api.polymarket.com`
   - Endpoint: `/markets`
   - Returns: List of active prediction markets

2. **CLOB API** (Order books)
   - Base: `https://clob.polymarket.com`
   - Endpoint: `/order-books`
   - Returns: Bid/ask prices for each outcome

### Rate Limits
- No official rate limits published
- Conservative: 1 request per market per cycle
- Cache market list for 5 minutes
- ~50 concurrent markets per cycle

---

## Troubleshooting

### "No market data available"
```
Possible causes:
1. Network connectivity issue
2. Polymarket API temporary downtime
3. Rate limiting (wait 60 seconds)

Solution:
- Check internet connection
- Try `python -m src.deploy_live test` to verify API
- Wait and retry
```

### "Model not found"
```
Error: Model not found at models/ensemble_model.h5

Solution:
- Run: python -m src.train_model
- Wait for training to complete (~5 min)
- Retry live deployment
```

### "Low predictions (0 BUY signals)"
```
Possible causes:
1. Markets too efficient (tight spreads)
2. Confidence threshold too high (75%)
3. Model uncertainty too high

Investigation:
- Check live prices: python -m src.deploy_live test
- Review market conditions
- Consider lowering threshold temporarily (for testing)
```

---

## Next Steps After Live Testing

### If A/B Test Wins (ML > Control)
1. ✅ Run for 2-4 weeks with real data
2. ✅ Deploy real capital ($500-2,000)
3. ✅ Scale positions based on proven edge
4. ✅ Monitor drawdowns + Sharpe ratio

### If Results Are Neutral
1. ⚠ Need more data (4-8 weeks min)
2. Improve model (retrain on more data)
3. Adjust thresholds
4. Review market conditions

### If Control Wins (Lottery > ML)
1. ❌ Debug model (overfitting?)
2. ❌ Check live API data quality
3. ❌ Review feature engineering
4. ❌ Possible model retraining needed

---

## Production Checklist

Before deploying real capital:

- [ ] Live API test passes (200+ markets fetched)
- [ ] Model predictions working (confidence + uncertainty)
- [ ] Paper trading runs 48+ hours without errors
- [ ] A/B test shows positive edge (>100 bps)
- [ ] Risk management working (no position over 10% wallet)
- [ ] Alerts/monitoring set up
- [ ] Database backup strategy ready
- [ ] Emergency stop procedure documented

---

## Configuration

Edit settings in `config/risk.json`:

```json
{
  "live_deployment": {
    "initial_balance": 50.0,
    "risk_per_trade": 0.05,
    "max_positions": 10,
    "confidence_threshold": 0.75,
    "max_uncertainty": 0.15
  },
  "api": {
    "market_limit": 100,
    "cache_duration_seconds": 300,
    "timeout_seconds": 10
  },
  "cycle": {
    "interval_seconds": 3600,
    "report_frequency": "hourly"
  }
}
```

---

## Files Modified (Phase 2)

```
src/
├── live_integration.py    (+400 lines) - API client + data streaming
└── deploy_live.py         (+200 lines) - Live deployment script
data/
├── wallet_test.json       - Test bot state
└── ab_test_results.json   - A/B test metrics
```

---

## Support

### Questions?
- Check README.md for architecture
- Review DEVELOPMENT.md for roadmap
- Test API: `python -m src.deploy_live test`

### Issues?
- Enable debug mode (add logging)
- Check Polymarket status page
- Verify network connectivity
- Review error logs in `logs/`

---

**Ready for live trading!** 🚀

Start with: `python -m src.deploy_live`
