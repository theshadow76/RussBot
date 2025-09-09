"""
Payout Checker - Check payouts for all assets before starting trading
"""

import asyncio
from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

def load_assets_from_file(filename: str) -> list:
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

async def check_payouts(ssid: str, assets: list, min_payout: float = 90.0):
    """Check payouts for all assets and return those above threshold"""
    try:
        api = PocketOptionAsync(ssid)
        await asyncio.sleep(5)
        
        print(f"🔍 Checking payouts for {len(assets)} assets...")
        print(f"💰 Minimum payout threshold: {min_payout}%")
        print("=" * 60)
        
        valid_assets = []
        all_payouts = {}
        
        # Get all payouts at once (more efficient)
        try:
            full_payout_data = await api.payout()
            print(f"📊 Received payout data for {len(full_payout_data)} total assets")
        except Exception as e:
            print(f"⚠️ Error getting bulk payout data: {e}")
            full_payout_data = {}
        
        # Check each asset
        for i, asset in enumerate(assets, 1):
            try:
                # Try to get from bulk data first
                if asset in full_payout_data:
                    payout = float(full_payout_data[asset])
                else:
                    # Fallback to individual request
                    payout_data = await api.payout(asset)
                    payout = float(payout_data) if payout_data else 0.0
                
                all_payouts[asset] = payout
                
                status = "✅" if payout > min_payout else "❌"
                print(f"{status} {asset:15} | {payout:6.1f}% | {'VALID' if payout > min_payout else 'SKIP'}")
                
                if payout > min_payout:
                    valid_assets.append(asset)
                
                # Progress indicator
                if i % 20 == 0:
                    print(f"    ... {i}/{len(assets)} checked")
                    
            except Exception as e:
                print(f"❌ {asset:15} | ERROR: {e}")
                all_payouts[asset] = 0.0
        
        print("=" * 60)
        print(f"📈 Summary:")
        print(f"   Total assets checked: {len(assets)}")
        print(f"   Valid assets (>{min_payout}%): {len(valid_assets)}")
        print(f"   Filtered out: {len(assets) - len(valid_assets)}")
        
        # Show top payouts
        sorted_payouts = sorted(all_payouts.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Top 10 Payouts:")
        for asset, payout in sorted_payouts[:10]:
            print(f"   {asset:15} | {payout:6.1f}%")
        
        # Save valid assets to file
        valid_assets_file = "valid_assets.txt"
        with open(valid_assets_file, 'w') as f:
            for asset in valid_assets:
                f.write(f"{asset}\n")
        
        print(f"\n💾 Valid assets saved to: {valid_assets_file}")
        
        return valid_assets
        
    except Exception as e:
        print(f"❌ Error checking payouts: {e}")
        return []

async def main():
    """Main payout checker function"""
    print("💰 Asset Payout Checker")
    print("=" * 30)
    
    ssid = input('Please enter your SSID: ')
    
    try:
        min_payout = float(input('Minimum payout percentage (default: 90): ') or 90)
    except ValueError:
        min_payout = 90.0
    
    # Load assets
    assets = load_assets_from_file('assets-otc.tested.txt')
    if not assets:
        print("❌ No assets loaded")
        return
    
    # Check payouts
    valid_assets = await check_payouts(ssid, assets, min_payout)
    
    if valid_assets:
        print(f"\n🎯 Ready to trade {len(valid_assets)} assets:")
        for asset in valid_assets[:20]:  # Show first 20
            print(f"   • {asset}")
        if len(valid_assets) > 20:
            print(f"   ... and {len(valid_assets) - 20} more")
    else:
        print("\n❌ No assets meet the payout criteria")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Payout check cancelled")
    except Exception as e:
        print(f"❌ Error: {e}")
