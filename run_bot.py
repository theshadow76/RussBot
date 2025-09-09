"""
Simple script to run the trading bot
Handles potential import issues and provides clear instructions
"""

import asyncio
import sys
import os

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import pandas as pd
        print("✅ pandas available")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy as np
        print("✅ numpy available") 
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
        from BinaryOptionsToolsV2.tracing import start_logs
        print("✅ BinaryOptionsToolsV2 available")
    except ImportError:
        missing_deps.append("BinaryOptionsToolsV2")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("\nTo install missing dependencies:")
        if "pandas" in missing_deps:
            print("  pip install pandas")
        if "numpy" in missing_deps:
            print("  pip install numpy")
        if "BinaryOptionsToolsV2" in missing_deps:
            print("  pip install BinaryOptionsToolsV2")
        return False
    
    print("✅ All dependencies available!")
    return True

def main():
    """Main execution function"""
    print("🤖 RussBot Trading Bot")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        return
    
    print("\nAvailable options:")
    print("1. Run Trading Bot (trading_bot.py)")
    print("2. Multi-Asset Bot (multi_asset_bot.py)")
    print("3. Check Asset Payouts (payout_checker.py)")
    print("4. Test Indicators (test_indicators.py)")
    print("5. Test Historical Data (test_history.py)")
    print("6. Simple History Test (test_simple_history.py)")
    print("7. View Configuration (config.py)")
    print("8. Exit")
    
    choice = input("\nSelect option (1-8): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Trading Bot...")
        try:
            # Import and run the trading bot
            import trading_bot
            asyncio.run(trading_bot.main())
        except Exception as e:
            print(f"❌ Error running trading bot: {e}")
    
    elif choice == "2":
        print("\n🔄 Starting Multi-Asset Bot...")
        try:
            import multi_asset_bot
            asyncio.run(multi_asset_bot.main())
        except Exception as e:
            print(f"❌ Error running multi-asset bot: {e}")
    
    elif choice == "3":
        print("\n💰 Starting Payout Checker...")
        try:
            import payout_checker
            asyncio.run(payout_checker.main())
        except Exception as e:
            print(f"❌ Error running payout checker: {e}")
    
    elif choice == "4":
        print("\n🧪 Starting Indicator Tests...")
        try:
            import test_indicators
            asyncio.run(test_indicators.main())
        except Exception as e:
            print(f"❌ Error running tests: {e}")
    
    elif choice == "5":
        print("\n📈 Starting Historical Data Test...")
        try:
            import test_history
            asyncio.run(test_history.main())
        except Exception as e:
            print(f"❌ Error running historical data test: {e}")
    
    elif choice == "6":
        print("\n🧪 Starting Simple History Test...")
        try:
            import test_simple_history
            asyncio.run(test_simple_history.main())
        except Exception as e:
            print(f"❌ Error running simple history test: {e}")
    
    elif choice == "7":
        print("\n⚙️ Configuration:")
        try:
            import config
            print("Trading Configuration:")
            for key, value in config.TRADING_CONFIG.items():
                print(f"  {key}: {value}")
            
            print("\nStrategy Rules:")
            print("Buy conditions:")
            for condition in config.STRATEGY_RULES["buy_conditions"]:
                print(f"  - {condition}")
            print("Sell conditions:")
            for condition in config.STRATEGY_RULES["sell_conditions"]:
                print(f"  - {condition}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
    
    elif choice == "8":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")
    
    input("\nPress Enter to continue...")
    main()  # Return to menu

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        input("Press Enter to exit...")
