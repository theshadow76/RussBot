"""
Test script for technical indicators
Tests EMA, CCI, and MACD calculations with sample data
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import asyncio
from datetime import timedelta

try:
    from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
    LIVE_TRADING_AVAILABLE = True
except ImportError:
    LIVE_TRADING_AVAILABLE = False
    print("⚠️ BinaryOptionsToolsV2 not available - testing indicators only")


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
            ema = (data[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
        
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
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(data)):
            if pd.isna(ema_fast[i]) or pd.isna(ema_slow[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Calculate Signal line (EMA of MACD)
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        # Calculate Histogram
        histogram = []
        for i in range(len(macd_line)):
            if pd.isna(macd_line[i]) or pd.isna(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram


def test_indicators_with_sample_data():
    """Test indicators with sample price data"""
    print("🧪 Testing Technical Indicators...")
    
    # Sample price data (simulating EURUSD prices)
    np.random.seed(42)  # For reproducible results
    base_price = 1.0800
    price_changes = np.random.normal(0, 0.0001, 50)
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] + change
        prices.append(new_price)
    
    # Create OHLC data from prices
    closes = prices
    highs = [price + abs(np.random.normal(0, 0.00005)) for price in prices]
    lows = [price - abs(np.random.normal(0, 0.00005)) for price in prices]
    
    print(f"📊 Sample data: {len(closes)} candles")
    print(f"Price range: {min(closes):.5f} - {max(closes):.5f}")
    
    # Calculate indicators
    ema_10 = TechnicalIndicators.ema(closes, 10)
    cci_7 = TechnicalIndicators.cci(highs, lows, closes, 7)
    macd_line, signal_line, histogram = TechnicalIndicators.macd(closes, 12, 26, 9)
    
    # Display last few values
    print("\n📈 Last 5 indicator values:")
    print("Index | Price  | EMA(10) | CCI(7)  | MACD   | Signal")
    print("-" * 55)
    
    for i in range(max(0, len(closes) - 5), len(closes)):
        price = closes[i]
        ema = ema_10[i] if not pd.isna(ema_10[i]) else None
        cci = cci_7[i] if not pd.isna(cci_7[i]) else None
        macd = macd_line[i] if not pd.isna(macd_line[i]) else None
        signal = signal_line[i] if not pd.isna(signal_line[i]) else None
        
        print(f"{i:5d} | {price:.5f} | {ema:.5f if ema else 'N/A':>7} | "
              f"{cci:.2f if cci else 'N/A':>6} | {macd:.5f if macd else 'N/A':>6} | "
              f"{signal:.5f if signal else 'N/A':>6}")
    
    # Check for potential signals
    print("\n🔍 Signal Analysis (last candle):")
    last_idx = len(closes) - 1
    
    if last_idx > 0:
        current_price = closes[last_idx]
        prev_price = closes[last_idx - 1]
        current_ema = ema_10[last_idx]
        current_cci = cci_7[last_idx]
        
        is_green = current_price > prev_price
        above_ema = current_price > current_ema if not pd.isna(current_ema) else False
        cci_near_100 = abs(current_cci - 100) <= 10 if not pd.isna(current_cci) else False
        cci_near_minus_100 = abs(current_cci + 100) <= 10 if not pd.isna(current_cci) else False
        
        print(f"Candle color: {'🟢 Green' if is_green else '🔴 Red'}")
        print(f"Price vs EMA: {'Above' if above_ema else 'Below'} EMA")
        print(f"CCI value: {current_cci:.2f if not pd.isna(current_cci) else 'N/A'}")
        print(f"CCI near +100: {'Yes' if cci_near_100 else 'No'}")
        print(f"CCI near -100: {'Yes' if cci_near_minus_100 else 'No'}")


async def test_live_data(ssid: str, asset: str = "EURUSD_otc"):
    """Test indicators with live market data"""
    print(f"\n🔴 Testing with LIVE data from {asset}...")
    
    try:
        api = PocketOptionAsync(ssid)
        await asyncio.sleep(5)
        
        # Get current balance
        balance = await api.balance()
        print(f"Account balance: {balance}")
        
        # Collect some live candles
        candles_collected = []
        stream = await api.subscribe_symbol_timed(asset, timedelta(seconds=15))
        
        print("📡 Collecting live candles... (will collect 5 candles)")
        
        async for candle in stream:
            candles_collected.append(candle)
            print(f"Candle {len(candles_collected)}: {candle}")
            
            if len(candles_collected) >= 5:
                break
        
        # Extract price data
        closes = [float(c.get('close', 0)) for c in candles_collected]
        highs = [float(c.get('high', c.get('close', 0))) for c in candles_collected]
        lows = [float(c.get('low', c.get('close', 0))) for c in candles_collected]
        
        print(f"\n📊 Collected {len(closes)} live candles")
        print("Prices:", [f"{p:.5f}" for p in closes])
        
        # Calculate indicators (need more data for meaningful results)
        if len(closes) >= 10:
            ema_10 = TechnicalIndicators.ema(closes, 10)
            cci_7 = TechnicalIndicators.cci(highs, lows, closes, 7)
            
            print("\nIndicators from live data:")
            print(f"Last EMA(10): {ema_10[-1]:.5f if not pd.isna(ema_10[-1]) else 'N/A'}")
            print(f"Last CCI(7): {cci_7[-1]:.2f if not pd.isna(cci_7[-1]) else 'N/A'}")
        else:
            print("Need more candles for meaningful indicator calculation")
            
    except Exception as e:
        print(f"❌ Error testing live data: {e}")


async def main():
    """Main test function"""
    print("🤖 Technical Indicators Test Suite")
    print("=" * 40)
    
    # Test with sample data
    test_indicators_with_sample_data()
    
    # Ask if user wants to test with live data
    test_live = input("\n🔴 Test with live market data? (y/n): ").lower().strip()
    
    if test_live == 'y':
        ssid = input("Enter your SSID: ")
        asset = input("Enter asset (default: EURUSD_otc): ").strip() or "EURUSD_otc"
        await test_live_data(ssid, asset)
    
    print("\n✅ Testing completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
