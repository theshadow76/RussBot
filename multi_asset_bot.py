"""
Multi-Asset Trading Bot with EMA, CCI, and MACD Strategy
Features:
- Multiprocessing for multiple assets
- Payout filtering (>90%)
- 1-second candle subscription
- All assets from assets-otc.tested.txt
"""

import asyncio
import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import timedelta
from collections import deque
from typing import List, Dict, Optional, Tuple, Set
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import time
import os
import signal
import sys

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


class MultiAssetTradingBot:
    """Multi-asset trading bot class"""
    
    def __init__(self, ssid: str, asset: str, amount: float = 1.0, process_id: int = 0):
        self.ssid = ssid
        self.asset = asset
        self.amount = amount
        self.process_id = process_id
        self.max_history = 100
        
        # Data storage
        self.candles_history = deque(maxlen=self.max_history)
        self.high_prices = deque(maxlen=self.max_history)
        self.low_prices = deque(maxlen=self.max_history)
        self.close_prices = deque(maxlen=self.max_history)
        self.open_prices = deque(maxlen=self.max_history)
        self.timestamps = deque(maxlen=self.max_history)
        
        # Previous values for trend detection
        self.prev_ema = None
        self.prev_cci = None
        
        # API instance
        self.api = None
        
        # Trading state
        self.last_trade_time = 0
        self.min_trade_interval = 16  # Minimum 16 seconds between trades
        self.payout = 0.0
        
    async def initialize(self):
        """Initialize the API connection and check payout"""
        try:
            self.api = PocketOptionAsync(self.ssid)
            
            # Wait longer for API to fully initialize
            print(f"[P{self.process_id}] {self.asset}: Connecting to API...")
            await asyncio.sleep(5)
            
            # Wait for assets to be initialized
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    # Test connection by getting payout
                    payout_data = await self.api.payout(self.asset)
                    if payout_data is not None:
                        self.payout = float(payout_data)
                        break
                except Exception as e:
                    if "not initialized" in str(e).lower():
                        print(f"[P{self.process_id}] {self.asset}: Waiting for assets to initialize... ({attempt+1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    else:
                        raise e
            else:
                print(f"[P{self.process_id}] {self.asset}: Failed to initialize after {max_retries} attempts")
                return False
            
            print(f"[P{self.process_id}] {self.asset}: Payout {self.payout:.1f}%")
            
            # Only proceed if payout > 90%
            if self.payout <= 90.0:
                print(f"[P{self.process_id}] {self.asset}: Payout too low ({self.payout:.1f}%), skipping")
                return False
            
            # Load historical data
            await self.load_historical_data()
            return True
            
        except Exception as e:
            print(f"[P{self.process_id}] {self.asset}: Initialization error: {e}")
            return False
    
    async def load_historical_data(self):
        """Load historical candle data for indicators initialization"""
        try:
            # Get historical candles using the history method
            history_candles = await self.api.history(self.asset, 3600)
            
            if not history_candles:
                print(f"[P{self.process_id}] {self.asset}: No historical data")
                return
            
            # Convert historical data to our format and add to storage
            valid_candles = 0
            for candle in history_candles:
                formatted_candle = self.format_candle_data(candle)
                if formatted_candle:
                    self.add_candle_data_silent(formatted_candle)
                    valid_candles += 1
            
            print(f"[P{self.process_id}] {self.asset}: Loaded {valid_candles} historical candles")
            
        except Exception as e:
            print(f"[P{self.process_id}] {self.asset}: Error loading historical data: {e}")
    
    def format_candle_data(self, candle: Dict) -> Optional[Dict]:
        """Format and validate candle data from API"""
        try:
            # Extract basic values
            open_price = float(candle.get('open', 0))
            high = float(candle.get('high', 0))
            low = float(candle.get('low', 0))
            close = float(candle.get('close', 0))
            
            # Validate basic data
            if any(price <= 0 for price in [open_price, high, low, close]):
                return None
            
            # Create list of all prices to find actual high/low
            all_prices = [open_price, close, high, low]
            actual_high = max(all_prices)
            actual_low = min(all_prices)
            
            return {
                'open': open_price,
                'high': actual_high,
                'low': actual_low,
                'close': close,
                'time': candle.get('time', 0)
            }
            
        except (ValueError, TypeError, KeyError):
            return None
    
    def add_candle_data_silent(self, candle: Dict):
        """Add new candle data to history without printing"""
        try:
            high = float(candle['high'])
            low = float(candle['low'])
            close = float(candle['close'])
            open_price = float(candle['open'])
            timestamp = candle.get('time', 0)
            
            self.high_prices.append(high)
            self.low_prices.append(low)
            self.close_prices.append(close)
            self.open_prices.append(open_price)
            self.timestamps.append(timestamp)
            self.candles_history.append(candle)
            
        except (ValueError, TypeError):
            pass
    
    def calculate_indicators(self) -> Optional[Dict]:
        """Calculate all technical indicators"""
        if len(self.close_prices) < 26:
            return None
        
        close_list = list(self.close_prices)
        high_list = list(self.high_prices)
        low_list = list(self.low_prices)
        
        # Calculate indicators
        ema_10 = TechnicalIndicators.ema(close_list, 10)
        cci_7 = TechnicalIndicators.cci(high_list, low_list, close_list, 7)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close_list, 12, 26, 9)
        
        # Get current values
        current_ema = ema_10[-1] if len(ema_10) > 0 and not pd.isna(ema_10[-1]) else None
        current_cci = cci_7[-1] if len(cci_7) > 0 and not pd.isna(cci_7[-1]) else None
        current_macd = macd_line[-1] if len(macd_line) > 0 and not pd.isna(macd_line[-1]) else None
        current_signal = signal_line[-1] if len(signal_line) > 0 and not pd.isna(signal_line[-1]) else None
        
        if any(val is None for val in [current_ema, current_cci, current_macd, current_signal]):
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
        current_close = float(current_candle['close'])
        current_open = float(current_candle['open'])
        current_ema = indicators['ema']
        current_cci = indicators['cci']
        
        # Check conditions
        is_green_candle = current_close > current_open
        price_above_ema = current_close > current_ema
        cci_touches_100 = abs(current_cci - 100) <= 10
        ema_uptrend = self.prev_ema is not None and current_ema > self.prev_ema
        
        return is_green_candle and price_above_ema and cci_touches_100 and ema_uptrend
    
    def check_sell_signal(self, indicators: Dict, current_candle: Dict) -> bool:
        """Check for sell signal conditions"""
        current_close = float(current_candle['close'])
        current_open = float(current_candle['open'])
        current_ema = indicators['ema']
        current_cci = indicators['cci']
        
        # Check conditions
        is_red_candle = current_close < current_open
        price_below_ema = current_close < current_ema
        cci_touches_minus_100 = abs(current_cci + 100) <= 10
        ema_downtrend = self.prev_ema is not None and current_ema < self.prev_ema
        
        return is_red_candle and price_below_ema and cci_touches_minus_100 and ema_downtrend
    
    def can_trade(self) -> bool:
        """Check if enough time has passed since last trade"""
        current_time = time.time()
        return (current_time - self.last_trade_time) >= self.min_trade_interval
    
    async def execute_buy_trade(self):
        """Execute a buy trade"""
        if not self.can_trade():
            return
        
        try:
            self.last_trade_time = time.time()
            print(f"[P{self.process_id}] 🟢 BUY {self.asset} @ {self.amount}")
            
            (trade_id, _) = await self.api.buy(
                asset=self.asset,
                amount=self.amount,
                time=15,
                check_win=False
            )
            
            print(f"[P{self.process_id}] ✅ Buy trade {trade_id} executed for {self.asset}")
            
        except Exception as e:
            print(f"[P{self.process_id}] ❌ Buy trade error for {self.asset}: {e}")
    
    async def execute_sell_trade(self):
        """Execute a sell trade"""
        if not self.can_trade():
            return
        
        try:
            self.last_trade_time = time.time()
            print(f"[P{self.process_id}] 🔴 SELL {self.asset} @ {self.amount}")
            
            (trade_id, _) = await self.api.sell(
                asset=self.asset,
                amount=self.amount,
                time=15,
                check_win=False
            )
            
            print(f"[P{self.process_id}] ✅ Sell trade {trade_id} executed for {self.asset}")
            
        except Exception as e:
            print(f"[P{self.process_id}] ❌ Sell trade error for {self.asset}: {e}")
    
    async def analyze_and_trade(self, candle: Dict):
        """Analyze current market conditions and execute trades"""
        formatted_candle = self.format_candle_data(candle)
        if not formatted_candle:
            return
        
        self.add_candle_data_silent(formatted_candle)
        
        # Calculate indicators
        indicators = self.calculate_indicators()
        if indicators is None:
            return
        
        # Check for trading signals
        if self.check_buy_signal(indicators, formatted_candle):
            await self.execute_buy_trade()
        elif self.check_sell_signal(indicators, formatted_candle):
            await self.execute_sell_trade()
        
        # Update previous values
        self.prev_ema = indicators['ema']
        self.prev_cci = indicators['cci']
    
    async def run(self):
        """Main bot execution loop"""
        try:
            print(f"[P{self.process_id}] 🚀 Starting {self.asset} (Payout: {self.payout:.1f}%)")
            
            # Subscribe to symbol with 1-second timed candles
            stream = await self.api.subscribe_symbol_timed(
                self.asset,
                timedelta(seconds=1)  # Changed to 1 second
            )
            
            async for candle in stream:
                try:
                    await self.analyze_and_trade(candle)
                except Exception as e:
                    print(f"[P{self.process_id}] ❌ Error processing {self.asset}: {e}")
                    continue
                    
        except Exception as e:
            print(f"[P{self.process_id}] ❌ Critical error for {self.asset}: {e}")


def load_assets_from_file(filename: str) -> List[str]:
    """Load assets from file, filter out commented lines"""
    assets = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    assets.append(line)
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
    return assets


async def run_single_asset(ssid: str, asset: str, amount: float, process_id: int):
    """Run trading bot for a single asset"""
    try:
        bot = MultiAssetTradingBot(ssid, asset, amount, process_id)
        
        # Initialize and check payout
        if await bot.initialize():
            await bot.run()
        else:
            print(f"[P{process_id}] {asset}: Skipped (initialization failed or low payout)")
            
    except Exception as e:
        print(f"[P{process_id}] {asset}: Fatal error: {e}")


def worker_process(ssid: str, asset: str, amount: float, process_id: int):
    """Worker process function"""
    try:
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"[P{process_id}] {asset}: Received shutdown signal")
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run the async function in the new loop
            loop.run_until_complete(run_single_asset(ssid, asset, amount, process_id))
        finally:
            loop.close()
        
    except KeyboardInterrupt:
        print(f"[P{process_id}] {asset}: Interrupted")
    except Exception as e:
        print(f"[P{process_id}] {asset}: Worker error: {e}")


async def main():
    """Main function to run multi-asset trading bot"""
    print("🤖 Multi-Asset Trading Bot")
    print("=" * 50)
    
    # Get user input
    ssid = input('Please enter your SSID: ')
    
    try:
        amount = float(input('Enter trade amount per asset (default: 1.0): ') or 1.0)
    except ValueError:
        amount = 1.0
    
    try:
        max_processes = int(input('Max concurrent processes (default: 5): ') or 5)
    except ValueError:
        max_processes = 5
    
    # Load assets from file
    assets = load_assets_from_file('assets-otc.tested.txt')
    if not assets:
        print("❌ No assets loaded from file")
        return
    
    print(f"📊 Loaded {len(assets)} assets from file")
    print(f"💰 Trade amount per asset: {amount}")
    print(f"🔄 Max concurrent processes: {max_processes}")
    print(f"📈 Payout filter: >90%")
    print(f"⏱️ Candle interval: 1 second")
    
    # Create and start processes
    processes = []
    active_assets = []
    
    try:
        for i, asset in enumerate(assets):
            if len(processes) >= max_processes:
                break
            
            print(f"🚀 Starting process {i+1} for {asset}")
            
            p = mp.Process(
                target=worker_process,
                args=(ssid, asset, amount, i+1)
            )
            p.start()
            processes.append(p)
            active_assets.append(asset)
            
            # Longer delay to avoid overwhelming the API
            await asyncio.sleep(2.0)
        
        print(f"\n✅ Started {len(processes)} processes for assets with >90% payout")
        print("🎯 Active assets:", ", ".join(active_assets[:10]) + ("..." if len(active_assets) > 10 else ""))
        print("\n📈 Monitoring trades... Press Ctrl+C to stop all processes")
        
        # Wait for all processes
        for p in processes:
            p.join()
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping all processes...")
        for p in processes:
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
        print("✅ All processes stopped")


if __name__ == '__main__':
    try:
        # Enable multiprocessing on Windows
        mp.set_start_method('spawn', force=True)
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Multi-asset bot terminated")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
