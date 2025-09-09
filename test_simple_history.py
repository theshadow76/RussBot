"""
Simple test script for the history function
Based on the provided example
"""

from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import pandas as pd
import asyncio

async def main(ssid: str):
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)    
    await asyncio.sleep(5)
    
    asset = "EURUSD_otc"
    period = 3600  # 1 hour
    
    print(f"📈 Testing history function for {asset} with {period}s period...")
    
    try:
        # Candles are returned in the format of a list of dictionaries
        candles = await api.history(asset, period)
        print(f"✅ Raw Candles received: {len(candles) if candles else 0}")
        
        if candles:
            # Show first few candles
            print(f"\n📊 First 3 candles structure:")
            for i, candle in enumerate(candles[:3]):
                print(f"  {i+1}: {candle}")
                
            # Check if any have high < low issue
            swapped_count = 0
            for candle in candles:
                high = float(candle.get('high', 0))
                low = float(candle.get('low', 0))
                if high < low:
                    swapped_count += 1
            
            print(f"\n⚠️ Candles with high < low: {swapped_count}/{len(candles)}")
            
            # Convert to pandas DataFrame
            candles_pd = pd.DataFrame.from_dict(candles)
            print(f"\n📈 Candles DataFrame shape: {candles_pd.shape}")
            print(f"Columns: {list(candles_pd.columns)}")
            
            if not candles_pd.empty:
                print(f"\n📊 Data sample:")
                print(candles_pd.head())
                
                print(f"\n📈 Price statistics:")
                if 'close' in candles_pd.columns:
                    print(f"  Close prices - Min: {candles_pd['close'].min():.5f}, Max: {candles_pd['close'].max():.5f}")
                if 'high' in candles_pd.columns:
                    print(f"  High prices - Min: {candles_pd['high'].min():.5f}, Max: {candles_pd['high'].max():.5f}")
                if 'low' in candles_pd.columns:
                    print(f"  Low prices - Min: {candles_pd['low'].min():.5f}, Max: {candles_pd['low'].max():.5f}")
                    
                # Check for data quality issues
                if 'high' in candles_pd.columns and 'low' in candles_pd.columns:
                    invalid_hl = (candles_pd['high'] < candles_pd['low']).sum()
                    print(f"  📊 Invalid high < low count: {invalid_hl}")
        else:
            print("❌ No candles received")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))
