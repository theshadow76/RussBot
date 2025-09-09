"""
Trading Bot with EMA, CCI, and MACD Strategy
Strategy: 
- Buy: Green candle crosses EMA up + CCI touches 100 + EMA crosses from down up
- Sell: Red candle crosses EMA down + CCI touches -100 + EMA crosses from up down
- Candle time: 15s, Expiry: 15s
- No martingale
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta
from collections import deque
from typing import List, Dict, Optional, Tuple
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync


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
        
        # Calculate MACD line
        macd_line = []
        for i in range(len(data)):
            if pd.isna(ema_fast[i]) or pd.isna(ema_slow[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])
        
        # Filter out NaN values for signal line calculation
        valid_macd_values = []
        valid_indices = []
        for i, val in enumerate(macd_line):
            if not pd.isna(val):
                valid_macd_values.append(val)
                valid_indices.append(i)
        
        # Calculate Signal line (EMA of MACD) only on valid values
        if len(valid_macd_values) >= signal:
            signal_ema = TechnicalIndicators.ema(valid_macd_values, signal)
            
            # Map back to full array
            signal_line = [np.nan] * len(data)
            for i, idx in enumerate(valid_indices):
                if i < len(signal_ema) and not pd.isna(signal_ema[i]):
                    signal_line[idx] = signal_ema[i]
        else:
            signal_line = [np.nan] * len(data)
        
        # Calculate Histogram
        histogram = []
        for i in range(len(macd_line)):
            if pd.isna(macd_line[i]) or pd.isna(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])
        
        return macd_line, signal_line, histogram


class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, ssid: str, asset: str = "EURUSD_otc", amount: float = 1.0, max_history: int = 100):
        self.ssid = ssid
        self.asset = asset
        self.amount = amount
        self.max_history = max_history
        
        # Data storage
        self.candles_history = deque(maxlen=max_history)
        self.high_prices = deque(maxlen=max_history)
        self.low_prices = deque(maxlen=max_history)
        self.close_prices = deque(maxlen=max_history)
        self.open_prices = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
        # Previous values for trend detection
        self.prev_ema = None
        self.prev_cci = None
        
        # API instance
        self.api = None
        
        # Trading state
        self.last_trade_time = 0
        self.min_trade_interval = 16  # Minimum 16 seconds between trades (expiry + 1s)
        
    async def initialize(self):
        """Initialize the API connection"""
        self.api = PocketOptionAsync(self.ssid)
        await asyncio.sleep(5)  # Wait for connection
        print("Trading bot initialized successfully")
        
        # Get initial balance
        balance = await self.api.balance()
        print(f"Current balance: {balance}")
        
        # Load historical data for indicators
        await self.load_historical_data()
    
    async def load_historical_data(self):
        """Load historical candle data for indicators initialization"""
        print(f"📈 Loading historical data for {self.asset}...")
        
        try:
            # Get historical candles using the history method
            # Get last 3600 seconds (1 hour) of data which should give us enough candles
            history_candles = await self.api.history(self.asset, 3600)
            
            if not history_candles:
                print("⚠️ No historical data received, starting with live data only")
                return
            
            print(f"📊 Received {len(history_candles)} historical candles")
            
            # Debug: Show first candle structure
            if history_candles:
                print(f"🔍 Sample candle structure: {history_candles[0]}")
            
            # Convert historical data to our format and add to storage
            valid_candles = 0
            for i, candle in enumerate(history_candles):
                # Convert candle format if needed
                formatted_candle = self.format_candle_data(candle)
                 
                if formatted_candle:
                    # Add to our data storage (without printing each one)
                    self.add_candle_data_silent(formatted_candle)
                    valid_candles += 1
                
                # Show progress every 20 candles
                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{len(history_candles)} candles...")
            
            print(f"✅ Loaded {valid_candles} valid candles into memory")
            
            # Debug: Show some sample data
            if len(self.close_prices) >= 5:
                print(f"🔍 Sample close prices: {list(self.close_prices)[-5:]}")
                print(f"🔍 Sample high prices: {list(self.high_prices)[-5:]}")
                print(f"🔍 Sample low prices: {list(self.low_prices)[-5:]}")
            
            # Calculate initial indicators to verify data
            indicators = self.calculate_indicators()
            if indicators:
                print(f"🔧 Initial indicators calculated:")
                print(f"   EMA(10): {indicators['ema']:.5f}")
                print(f"   CCI(7): {indicators['cci']:.2f}")
                print(f"   MACD: {indicators['macd']:.5f}")
                print("🟢 Ready for live trading!")
            else:
                print("⚠️ Could not calculate indicators yet, need more data")
                
        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            print("⚠️ Starting without historical data - will wait for live candles")
    
    def format_candle_data(self, candle: Dict) -> Optional[Dict]:
        """Format and validate candle data from API"""
        try:
            # Extract basic values
            raw_open = candle.get('open', 0)
            raw_high = candle.get('high', 0) 
            raw_low = candle.get('low', 0)
            raw_close = candle.get('close', 0)
            raw_time = candle.get('time', 0)
            
            # Convert to floats
            open_price = float(raw_open)
            high = float(raw_high)
            low = float(raw_low)
            close = float(raw_close)
            
            # Validate basic data
            if any(price <= 0 for price in [open_price, high, low, close]):
                return None
            
            # Create list of all prices to find actual high/low
            all_prices = [open_price, close, high, low]
            actual_high = max(all_prices)
            actual_low = min(all_prices)
            
            # Use actual high/low instead of potentially mislabeled ones
            formatted_candle = {
                'open': open_price,
                'high': actual_high,
                'low': actual_low,
                'close': close,
                'time': raw_time
            }
            
            return formatted_candle
            
        except (ValueError, TypeError, KeyError) as e:
            print(f"⚠️ Error formatting candle: {e}")
            return None
    
    def add_candle_data_silent(self, candle: Dict):
        """Add new candle data to history without printing (for historical data)"""
        # Extract OHLC data (should already be validated by format_candle_data)
        try:
            high = float(candle.get('high', 0))
            low = float(candle.get('low', 0))
            close = float(candle.get('close', 0))
            open_price = float(candle.get('open', 0))
            timestamp = candle.get('time', 0)
            
            # Store data
            self.high_prices.append(high)
            self.low_prices.append(low)
            self.close_prices.append(close)
            self.open_prices.append(open_price)
            self.timestamps.append(timestamp)
            self.candles_history.append(candle)
            
        except (ValueError, TypeError) as e:
            print(f"⚠️ Error processing candle data: {e}")
    
    def add_candle_data(self, candle: Dict):
        """Add new candle data to history"""
        # Format and validate the candle first
        formatted_candle = self.format_candle_data(candle)
        
        if not formatted_candle:
            print(f"⚠️ Skipping invalid candle: {candle}")
            return
        
        # Use silent method and then print
        self.add_candle_data_silent(formatted_candle)
        
        # Print the new candle info
        open_price = formatted_candle['open']
        high = formatted_candle['high']
        low = formatted_candle['low']
        close = formatted_candle['close']
        print(f"Added candle: O:{open_price:.5f} H:{high:.5f} L:{low:.5f} C:{close:.5f}")
    
    def calculate_indicators(self) -> Optional[Dict]:
        """Calculate all technical indicators"""
        if len(self.close_prices) < 26:  # Need at least 26 candles for MACD
            print(f"📊 Need more data: {len(self.close_prices)}/26 candles available")
            return None
        
        close_list = list(self.close_prices)
        high_list = list(self.high_prices)
        low_list = list(self.low_prices)
        
        print(f"🔍 Debug: Calculating indicators with {len(close_list)} candles")
        print(f"    Last 3 close prices: {close_list[-3:]}")
        
        # Calculate indicators
        ema_10 = TechnicalIndicators.ema(close_list, 10)
        cci_7 = TechnicalIndicators.cci(high_list, low_list, close_list, 7)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close_list, 12, 26, 9)
        
        # Get current values (last in the list)
        current_ema = ema_10[-1] if len(ema_10) > 0 and not pd.isna(ema_10[-1]) else None
        current_cci = cci_7[-1] if len(cci_7) > 0 and not pd.isna(cci_7[-1]) else None
        current_macd = macd_line[-1] if len(macd_line) > 0 and not pd.isna(macd_line[-1]) else None
        current_signal = signal_line[-1] if len(signal_line) > 0 and not pd.isna(signal_line[-1]) else None
        
        print(f"🔍 Debug indicator values:")
        print(f"    EMA(10): {current_ema}")
        print(f"    CCI(7): {current_cci}")
        print(f"    MACD: {current_macd}")
        print(f"    Signal: {current_signal}")
        
        if any(val is None for val in [current_ema, current_cci, current_macd, current_signal]):
            print("⚠️ Some indicator values are invalid")
            # Let's see which ones are None
            if current_ema is None:
                print("  - EMA is None")
            if current_cci is None:
                print("  - CCI is None")
            if current_macd is None:
                print("  - MACD is None")
            if current_signal is None:
                print("  - Signal is None")
            return None
        
        return {
            'ema': current_ema,
            'cci': current_cci,
            'macd': current_macd,
            'signal': current_signal,
            'ema_history': ema_10,
            'cci_history': cci_7
        }
    
    def check_buy_signal(self, indicators: Dict, current_candle: Dict) -> bool:
        """Check for buy signal conditions"""
        current_close = float(current_candle.get('close', 0))
        current_open = float(current_candle.get('open', current_close))
        current_ema = indicators['ema']
        current_cci = indicators['cci']
        
        # Check if current candle is green
        is_green_candle = current_close > current_open
        
        # Check if price crosses EMA up (close above EMA)
        price_above_ema = current_close > current_ema
        
        # Check if CCI touches 100 (within tolerance)
        cci_touches_100 = abs(current_cci - 100) <= 10  # 10 point tolerance
        
        # Check EMA trend (crosses from down to up)
        ema_uptrend = False
        if self.prev_ema is not None:
            ema_uptrend = current_ema > self.prev_ema
        
        print(f"Buy Check - Green: {is_green_candle}, Above EMA: {price_above_ema}, "
              f"CCI~100: {cci_touches_100} (CCI: {current_cci:.2f}), EMA Up: {ema_uptrend}")
        
        return is_green_candle and price_above_ema and cci_touches_100 and ema_uptrend
    
    def check_sell_signal(self, indicators: Dict, current_candle: Dict) -> bool:
        """Check for sell signal conditions"""
        current_close = float(current_candle.get('close', 0))
        current_open = float(current_candle.get('open', current_close))
        current_ema = indicators['ema']
        current_cci = indicators['cci']
        
        # Check if current candle is red
        is_red_candle = current_close < current_open
        
        # Check if price crosses EMA down (close below EMA)
        price_below_ema = current_close < current_ema
        
        # Check if CCI touches -100 (within tolerance)
        cci_touches_minus_100 = abs(current_cci + 100) <= 10  # 10 point tolerance
        
        # Check EMA trend (crosses from up to down)
        ema_downtrend = False
        if self.prev_ema is not None:
            ema_downtrend = current_ema < self.prev_ema
        
        print(f"Sell Check - Red: {is_red_candle}, Below EMA: {price_below_ema}, "
              f"CCI~-100: {cci_touches_minus_100} (CCI: {current_cci:.2f}), EMA Down: {ema_downtrend}")
        
        return is_red_candle and price_below_ema and cci_touches_minus_100 and ema_downtrend
    
    def can_trade(self) -> bool:
        """Check if enough time has passed since last trade"""
        import time
        current_time = time.time()
        return (current_time - self.last_trade_time) >= self.min_trade_interval
    
    async def execute_buy_trade(self):
        """Execute a buy trade"""
        if not self.can_trade():
            print("Cannot trade yet - waiting for cooldown period")
            return
        
        try:
            import time
            self.last_trade_time = time.time()
            
            print(f"🟢 EXECUTING BUY TRADE - Asset: {self.asset}, Amount: {self.amount}")
            (trade_id, trade_data) = await self.api.buy(
                asset=self.asset, 
                amount=self.amount, 
                time=15, 
                check_win=False
            )
            print(f"✅ Buy trade executed - ID: {trade_id}")
            
            # Optionally check win after expiry
            asyncio.create_task(self.check_trade_result(trade_id, "BUY"))
            
        except Exception as e:
            print(f"❌ Error executing buy trade: {e}")
    
    async def execute_sell_trade(self):
        """Execute a sell trade"""
        if not self.can_trade():
            print("Cannot trade yet - waiting for cooldown period")
            return
        
        try:
            import time
            self.last_trade_time = time.time()
            
            print(f"🔴 EXECUTING SELL TRADE - Asset: {self.asset}, Amount: {self.amount}")
            (trade_id, trade_data) = await self.api.sell(
                asset=self.asset, 
                amount=self.amount, 
                time=15, 
                check_win=False
            )
            print(f"✅ Sell trade executed - ID: {trade_id}")
            
            # Optionally check win after expiry
            asyncio.create_task(self.check_trade_result(trade_id, "SELL"))
            
        except Exception as e:
            print(f"❌ Error executing sell trade: {e}")
    
    async def check_trade_result(self, trade_id: str, trade_type: str):
        """Check trade result after expiry"""
        try:
            # Wait for trade to expire (15s + 2s buffer)
            await asyncio.sleep(17)
            
            result = await self.api.check_win(trade_id)
            status = result.get('result', 'unknown')
            
            if status == 'win':
                print(f"🎉 {trade_type} Trade {trade_id} WON!")
            elif status == 'loss':
                print(f"😞 {trade_type} Trade {trade_id} LOST")
            else:
                print(f"⏳ {trade_type} Trade {trade_id} status: {status}")
                
        except Exception as e:
            print(f"❌ Error checking trade {trade_id} result: {e}")
    
    async def analyze_and_trade(self, candle: Dict):
        """Analyze current market conditions and execute trades if signals are met"""
        self.add_candle_data(candle)
        
        # Calculate indicators
        indicators = self.calculate_indicators()
        if indicators is None:
            print("Not enough data for indicators yet...")
            return
        
        # Print current market state
        current_close = float(candle.get('close', 0))
        print(f"\n📊 Market Analysis:")
        print(f"Price: {current_close:.5f}")
        print(f"EMA(10): {indicators['ema']:.5f}")
        print(f"CCI(7): {indicators['cci']:.2f}")
        print(f"MACD: {indicators['macd']:.5f}")
        
        # Check for trading signals
        if self.check_buy_signal(indicators, candle):
            await self.execute_buy_trade()
        elif self.check_sell_signal(indicators, candle):
            await self.execute_sell_trade()
        else:
            print("No trading signals detected")
        
        # Update previous values for next iteration
        self.prev_ema = indicators['ema']
        self.prev_cci = indicators['cci']
    
    async def run(self):
        """Main bot execution loop"""
        print(f"🚀 Starting trading bot for {self.asset}")
        print(f"Strategy: EMA(10), CCI(7), MACD(12,26,9)")
        print(f"Candle time: 15s, Expiry: 15s")
        print("=" * 50)
        
        try:
            # Subscribe to symbol with 15-second timed candles
            stream = await self.api.subscribe_symbol_timed(
                self.asset, 
                timedelta(seconds=15)
            )
            
            async for candle in stream:
                try:
                    await self.analyze_and_trade(candle)
                except Exception as e:
                    print(f"❌ Error processing candle: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Critical error in bot execution: {e}")
        except KeyboardInterrupt:
            print("\n⏹️ Bot stopped by user")


async def main():
    """Main function to run the trading bot"""

    # Get user input
    ssid = input('Please enter your SSID: ')
    asset = input('Enter asset (default: EURUSD_otc): ').strip() or "EURUSD_otc"
    
    try:
        amount = float(input('Enter trade amount (default: 1.0): ') or 1.0)
    except ValueError:
        amount = 1.0
    
    print(f"\n🤖 Initializing trading bot...")
    print(f"Asset: {asset}")
    print(f"Amount: {amount}")
    
    # Create and run bot
    bot = TradingBot(ssid, asset, amount)
    await bot.initialize()
    
    # Check if we have enough data before starting
    if len(bot.close_prices) >= 26:
        print("🟢 Historical data loaded - Bot ready for trading!")
    else:
        print("🟡 Starting with live data only - will wait for enough candles")
    
    await bot.run()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Trading bot terminated")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
