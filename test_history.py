"""
Test script to verify historical data loading
This script tests the get_candles function to ensure it works correctly
"""

import asyncio
import sys
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

async def test_historical_data(ssid: str, asset: str = "EURUSD_otc"):
    """Test the historical data loading"""
    print(f"🧪 Testing historical data loading for {asset}")
    print("=" * 50)
    
    try:
        # Initialize API
        api = PocketOptionAsync(ssid)
        await asyncio.sleep(5)
        
        print("✅ API connection established")
        
        # Test balance
        balance = await api.balance()
        print(f"💰 Account balance: {balance}")
        
        # Test historical data loading
        print(f"\n📈 Loading historical candles...")
        
        # Test different history periods
        history_periods = [1800, 3600, 7200]  # 30min, 1h, 2h
        
        for period in history_periods:
            print(f"\n🔍 Testing {period}s history ({period//60} minutes):")
            
            try:
                candles = await api.history(asset, period)
                
                if candles:
                    print(f"  ✅ Received {len(candles)} candles")
                    
                    # Show first and last candle
                    first_candle = candles[0]
                    last_candle = candles[-1]
                    
                    print(f"  📊 First candle: O:{first_candle.get('open', 0):.5f} "
                          f"H:{first_candle.get('high', 0):.5f} "
                          f"L:{first_candle.get('low', 0):.5f} "
                          f"C:{first_candle.get('close', 0):.5f}")
                    
                    print(f"  📊 Last candle:  O:{last_candle.get('open', 0):.5f} "
                          f"H:{last_candle.get('high', 0):.5f} "
                          f"L:{last_candle.get('low', 0):.5f} "
                          f"C:{last_candle.get('close', 0):.5f}")
                    
                    # Validate data
                    valid_candles = 0
                    for candle in candles:
                        if all(key in candle for key in ['open', 'high', 'low', 'close']):
                            valid_candles += 1
                    
                    print(f"  ✅ Valid candles: {valid_candles}/{len(candles)}")
                    
                    if valid_candles >= 26:
                        print(f"  🟢 Sufficient data for indicators!")
                    else:
                        print(f"  🟡 Need more data for indicators (minimum 26)")
                
                else:
                    print(f"  ❌ No candles received")
                    
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print(f"\n🎯 Recommended settings for {asset}:")
        print(f"  History period: 3600s (1-hour)")
        print(f"  Expected candles: Variable (depends on market activity)")
        print(f"  Minimum needed: 26 candles for MACD")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
    print("\n✅ Historical data test completed!")

async def main():
    """Main test function"""
    print("🧪 Historical Data Test")
    print("=" * 30)
    
    ssid = input("Enter your SSID: ")
    asset = input("Enter asset (default: EURUSD_otc): ").strip() or "EURUSD_otc"
    
    await test_historical_data(ssid, asset)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Test interrupted")
    except Exception as e:
        print(f"❌ Test error: {e}")
