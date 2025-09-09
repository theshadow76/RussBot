from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync

import asyncio
# Main part of the code
async def main(ssid: str):
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)    
    await asyncio.sleep(5)
    
    single_payout = await api.payout("EURUSD_otc") # Returns the payout for the specified asset
    print(f"Single Payout: {single_payout}")
    
    
if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))
    