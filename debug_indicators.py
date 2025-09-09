"""
Debug script to test indicator calculations with sample data
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

class TechnicalIndicators:
    """Class to calculate technical indicators"""
    
    @staticmethod
    def ema(data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return [np.nan] * len(data)
        
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # Start with SMA for first value
        sma = sum(data[:period]) / period
        ema_values.extend([np.nan] * (period - 1))
        ema_values.append(sma)
        
        # Calculate EMA for remaining values
        for i in range(period, len(data)):
            try:
                ema = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
                ema_values.append(ema)
            except (TypeError, ValueError) as e:
                print(f"⚠️ EMA calculation error at index {i}: {e}")
                ema_values.append(np.nan)
        
        return ema_values

    @staticmethod
    def cci(high: List[float], low: List[float], close: List[float], period: int = 7) -> List[float]:
        """Calculate Commodity Channel Index"""
        if len(close) < period:
            return [np.nan] * len(close)
        
        cci_values = []
        
        for i in range(len(close)):
            if i < period - 1:
                cci_values.append(np.nan)
                continue
            
            # Calculate typical price for the period
            typical_prices = []
            for j in range(i - period + 1, i + 1):
                tp = (high[j] + low[j] + close[j]) / 3
                typical_prices.append(tp)
            
            # Calculate moving average of typical price
            sma_tp = sum(typical_prices) / period
            
            # Calculate mean absolute deviation
            mad = sum(abs(tp - sma_tp) for tp in typical_prices) / period
            
            # Calculate CCI
            current_tp = (high[i] + low[i] + close[i]) / 3
            if mad != 0:
                cci = (current_tp - sma_tp) / (0.015 * mad)
            else:
                cci = 0
            
            cci_values.append(cci)
        
        return cci_values

    @staticmethod
    def macd(data: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(data) < slow:
            return ([np.nan] * len(data), [np.nan] * len(data), [np.nan] * len(data))
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        
        print(f"Debug MACD: Fast EMA last 5: {ema_fast[-5:]}")
        print(f"Debug MACD: Slow EMA last 5: {ema_slow[-5:]}")
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(data)):
            if pd.isna(ema_fast[i]) or pd.isna(ema_slow[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        print(f"Debug MACD: MACD line last 5: {macd_line[-5:]}")
        
        # Filter out NaN values for signal line calculation
        valid_macd_values = []
        valid_indices = []
        for i, val in enumerate(macd_line):
            if not pd.isna(val):
                valid_macd_values.append(val)
                valid_indices.append(i)
        
        print(f"Debug MACD: Valid MACD values count: {len(valid_macd_values)}")
        print(f"Debug MACD: Signal period needed: {signal}")
        
        # Calculate Signal line (EMA of MACD) only on valid values
        if len(valid_macd_values) >= signal:
            signal_ema = TechnicalIndicators.ema(valid_macd_values, signal)
            print(f"Debug MACD: Signal EMA last 5: {signal_ema[-5:]}")
            
            # Map back to full array
            signal_line = [np.nan] * len(data)
            for i, idx in enumerate(valid_indices):
                if i < len(signal_ema) and not pd.isna(signal_ema[i]):
                    signal_line[idx] = signal_ema[i]
        else:
            signal_line = [np.nan] * len(data)
            print(f"Debug MACD: Not enough valid MACD values for signal line")
        
        print(f"Debug MACD: Signal line last 5: {signal_line[-5:]}")
        
        # Calculate Histogram
        histogram = []
        for i in range(len(macd_line)):
            if pd.isna(macd_line[i]) or pd.isna(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram

def test_indicators():
    print("🧪 Testing indicators with sample data...")
    
    # Generate sample price data similar to EURUSD
    np.random.seed(42)
    base_price = 1.18500
    price_changes = np.random.normal(0, 0.0001, 100)
    close_prices = [base_price]
    
    for change in price_changes:
        new_price = close_prices[-1] + change
        close_prices.append(new_price)
    
    # Create OHLC data
    high_prices = [price + abs(np.random.normal(0, 0.00005)) for price in close_prices]
    low_prices = [price - abs(np.random.normal(0, 0.00005)) for price in close_prices]
    
    print(f"Generated {len(close_prices)} price points")
    print(f"Close price range: {min(close_prices):.5f} - {max(close_prices):.5f}")
    print(f"Last 5 close prices: {close_prices[-5:]}")
    
    # Test EMA
    print("\n📈 Testing EMA(10)...")
    ema_10 = TechnicalIndicators.ema(close_prices, 10)
    print(f"EMA last value: {ema_10[-1] if not pd.isna(ema_10[-1]) else 'NaN'}")
    
    # Test CCI
    print("\n📊 Testing CCI(7)...")
    cci_7 = TechnicalIndicators.cci(high_prices, low_prices, close_prices, 7)
    print(f"CCI last value: {cci_7[-1] if not pd.isna(cci_7[-1]) else 'NaN'}")
    
    # Test MACD
    print("\n📉 Testing MACD(12,26,9)...")
    macd_line, signal_line, histogram = TechnicalIndicators.macd(close_prices, 12, 26, 9)
    print(f"MACD last value: {macd_line[-1] if not pd.isna(macd_line[-1]) else 'NaN'}")
    print(f"Signal last value: {signal_line[-1] if not pd.isna(signal_line[-1]) else 'NaN'}")
    print(f"Histogram last value: {histogram[-1] if not pd.isna(histogram[-1]) else 'NaN'}")
    
    # Check if all indicators are valid
    all_valid = (
        not pd.isna(ema_10[-1]) and 
        not pd.isna(cci_7[-1]) and 
        not pd.isna(macd_line[-1]) and 
        not pd.isna(signal_line[-1])
    )
    
    print(f"\n✅ All indicators valid: {all_valid}")

if __name__ == "__main__":
    test_indicators()
