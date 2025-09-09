# Trading Bot Configuration
TRADING_CONFIG = {
    # Asset settings
    "asset": "EURUSD_otc",
    "amount": 1.0,
    "expiry_time": 15,  # seconds
    
    # Indicator settings
    "ema_period": 10,
    "cci_period": 7,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    
    # Trading rules
    "cci_buy_threshold": 100,
    "cci_sell_threshold": -100,
    "cci_tolerance": 10,  # tolerance for CCI threshold
    "min_trade_interval": 16,  # minimum seconds between trades
    
    # Data management
    "max_history": 100,  # maximum candles to keep in memory
    
    # Logging
    "log_level": "INFO",
    "log_to_file": True,
    "log_to_terminal": True
}

# Strategy validation rules
STRATEGY_RULES = {
    "buy_conditions": [
        "Green candle (close > open)",
        "Price crosses EMA up (close > EMA)",
        "CCI touches 100 (±10 tolerance)",
        "EMA crosses from down up (uptrend)"
    ],
    "sell_conditions": [
        "Red candle (close < open)",
        "Price crosses EMA down (close < EMA)", 
        "CCI touches -100 (±10 tolerance)",
        "EMA crosses from up down (downtrend)"
    ]
}
