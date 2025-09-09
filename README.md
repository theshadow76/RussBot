# RussBot - Binary Options Trading Bot

A sophisticated trading bot implementing a strategy based on EMA, CCI, and MACD indicators for 15-second binary options trading.

## Strategy Overview

**Trading Setup:**
- **Timeframe:** 15-second candles
- **Expiry:** 15 seconds
- **Indicators:** 
  - EMA (10 periods)
  - CCI (7 periods) 
  - MACD (12, 26, 9)

**Buy Signal Conditions:**
1. 🟢 Green candle (close > open)
2. 📈 Price crosses EMA upward (close > EMA)
3. 📊 CCI touches +100 (±10 tolerance)
4. 🔄 EMA trending upward

**Sell Signal Conditions:**
1. 🔴 Red candle (close < open)
2. 📉 Price crosses EMA downward (close < EMA)
3. 📊 CCI touches -100 (±10 tolerance)
4. 🔄 EMA trending downward

**Risk Management:**
- No martingale system
- Minimum 16-second interval between trades
- Trade amount configurable

## Files Structure

```
RussBot/
├── trading_bot.py      # Main trading bot with full strategy
├── test_indicators.py  # Test script for indicators
├── config.py          # Configuration settings
├── run_bot.py         # Simple launcher script
├── context.txt        # API usage examples
└── README.md          # This file
```

## Installation & Requirements

### Required Dependencies
```bash
pip install pandas numpy BinaryOptionsToolsV2
```

### Python Version
- Python 3.7 or higher
- Asyncio support required

## Usage

### Option 1: Direct Execution
```bash
python trading_bot.py
```

### Option 2: Using Launcher
```bash
python run_bot.py
```

### Option 3: Test Indicators First
```bash
python test_indicators.py
```

## Configuration

Edit `config.py` to customize trading parameters:

```python
TRADING_CONFIG = {
    "asset": "EURUSD_otc",
    "amount": 1.0,
    "expiry_time": 15,
    "ema_period": 10,
    "cci_period": 7,
    "cci_tolerance": 10,
    # ... more settings
}
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install pandas numpy BinaryOptionsToolsV2
   ```

2. **Get your SSID:**
   - Log into your PocketOption account
   - Extract the SSID from browser cookies/network tab

3. **Test indicators first:**
   ```bash
   python test_indicators.py
   ```

4. **Run the bot:**
   ```bash
   python trading_bot.py
   ```

5. **Enter your SSID when prompted**

## Features

### Technical Indicators
- **EMA (Exponential Moving Average):** Trend direction
- **CCI (Commodity Channel Index):** Overbought/oversold conditions
- **MACD:** Momentum confirmation

### Trading Logic
- Real-time candle analysis using `subscribe_symbol_timed`
- Strict signal validation (all conditions must be met)
- Automatic trade execution with result tracking
- Comprehensive logging and error handling

### Safety Features
- Trade cooldown period (prevents overtrading)
- Input validation and error handling
- Position size management
- Real-time balance monitoring

## Strategy Explanation

The bot implements a multi-indicator confluence strategy:

1. **Trend Identification:** EMA determines the overall trend direction
2. **Entry Timing:** CCI extreme levels (±100) indicate potential reversal points
3. **Confirmation:** Candle color and price-EMA relationship confirm direction
4. **Execution:** All conditions must align for trade execution

## Monitoring & Logging

The bot provides real-time feedback:
- 📊 Market analysis for each candle
- 🔍 Signal detection status
- 💰 Trade execution confirmations
- 📈 Trade results after expiry

## Risk Disclaimer

**⚠️ IMPORTANT:** This bot is for educational purposes. Binary options trading carries significant financial risk. Never trade with money you cannot afford to lose.

- Test thoroughly on demo accounts first
- Start with small amounts
- Monitor bot performance closely
- Understand the strategy before using

## Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   pip install --upgrade BinaryOptionsToolsV2
   ```

2. **Connection Issues:**
   - Verify SSID is correct
   - Check internet connection
   - Ensure PocketOption account is active

3. **No Trading Signals:**
   - Strategy requires specific market conditions
   - All indicators must align simultaneously
   - Consider adjusting CCI tolerance in config

### Support

- Check `context.txt` for API usage examples
- Test individual indicators with `test_indicators.py`
- Review configuration in `config.py`

## License

This project is for educational purposes. Use at your own risk.

---

**Happy Trading! 🚀**
