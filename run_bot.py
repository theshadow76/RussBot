"""
RussBot Trading Bot Launcher
Simplified interface with 3 main options
"""

import asyncio
import sys
import os
import json
import webbrowser
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Load configuration from config.json"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ config.json not found, creating default configuration...")
        default_config = {
            "trading": {
                "default_asset": "EURUSD_otc",
                "default_amount": 1.0,
                "expiry_time": 15,
                "candle_interval": 1,
                "min_payout_percentage": 90.0,
                "max_concurrent_processes": 10,
                "min_trade_interval": 16
            },
            "indicators": {
                "ema_period": 10,
                "cci_period": 7,
                "cci_buy_threshold": 100,
                "cci_sell_threshold": -100,
                "cci_tolerance": 10,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9
            },
            "system": {
                "max_history": 100,
                "log_level": "INFO",
                "log_to_file": True,
                "log_to_terminal": True,
                "assets_file": "assets-otc.tested.txt"
            },
            "user": {
                "ssid": "",
                "last_used_asset": "EURUSD_otc",
                "last_used_amount": 1.0
            }
        }
        save_config(default_config)
        return default_config
    except json.JSONDecodeError:
        print("❌ Error reading config.json - invalid JSON format")
        return {}

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to config.json"""
    try:
        with open('config.json', 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import pandas as pd
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
        from BinaryOptionsToolsV2.tracing import start_logs
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
    
    return True

def start_bot():
    """Start the multi-asset trading bot"""
    print("\n🚀 Starting Multi-Asset Trading Bot...")
    print("=" * 50)
    
    if not check_dependencies():
        print("\n❌ Cannot start bot - missing dependencies")
        input("Press Enter to continue...")
        return
    
    try:
        import multi_asset_bot
        asyncio.run(multi_asset_bot.main())
    except KeyboardInterrupt:
        print("\n⏹️ Bot stopped by user")
    except Exception as e:
        print(f"❌ Error running multi-asset bot: {e}")
    
    input("\nPress Enter to return to menu...")

def configure_bot():
    """Configure bot settings"""
    print("\n⚙️ Bot Configuration")
    print("=" * 30)
    
    config = load_config()
    if not config:
        print("❌ Could not load configuration")
        input("Press Enter to continue...")
        return
    
    while True:
        print("\nConfiguration Options:")
        print("1. Trading Settings")
        print("2. Indicator Settings") 
        print("3. System Settings")
        print("4. User Settings")
        print("5. View Full Config")
        print("6. Reset to Defaults")
        print("7. Back to Main Menu")
        
        choice = input("\nSelect option (1-7): ").strip()
        
        if choice == "1":
            configure_trading_settings(config)
        elif choice == "2":
            configure_indicator_settings(config)
        elif choice == "3":
            configure_system_settings(config)
        elif choice == "4":
            configure_user_settings(config)
        elif choice == "5":
            view_full_config(config)
        elif choice == "6":
            if input("⚠️ Reset all settings to defaults? (y/N): ").lower() == 'y':
                config = load_config()  # This will create defaults if file is missing
                print("✅ Configuration reset to defaults")
        elif choice == "7":
            break
        else:
            print("❌ Invalid choice")

def configure_trading_settings(config: Dict[str, Any]):
    """Configure trading-specific settings"""
    print("\n💰 Trading Settings:")
    
    trading = config.get("trading", {})
    
    print(f"Current default asset: {trading.get('default_asset', 'N/A')}")
    new_asset = input("Enter new default asset (or press Enter to keep current): ").strip()
    if new_asset:
        trading["default_asset"] = new_asset
    
    print(f"Current default amount: {trading.get('default_amount', 'N/A')}")
    try:
        new_amount = input("Enter new default amount (or press Enter to keep current): ").strip()
        if new_amount:
            trading["default_amount"] = float(new_amount)
    except ValueError:
        print("⚠️ Invalid amount, keeping current value")
    
    print(f"Current min payout percentage: {trading.get('min_payout_percentage', 'N/A')}%")
    try:
        new_payout = input("Enter new min payout percentage (or press Enter to keep current): ").strip()
        if new_payout:
            trading["min_payout_percentage"] = float(new_payout)
    except ValueError:
        print("⚠️ Invalid percentage, keeping current value")
    
    print(f"Current max concurrent processes: {trading.get('max_concurrent_processes', 'N/A')}")
    try:
        new_processes = input("Enter new max concurrent processes (or press Enter to keep current): ").strip()
        if new_processes:
            trading["max_concurrent_processes"] = int(new_processes)
    except ValueError:
        print("⚠️ Invalid number, keeping current value")
    
    config["trading"] = trading
    if save_config(config):
        print("✅ Trading settings saved")
    else:
        print("❌ Error saving settings")

def configure_indicator_settings(config: Dict[str, Any]):
    """Configure indicator settings"""
    print("\n📊 Indicator Settings:")
    
    indicators = config.get("indicators", {})
    
    print(f"Current EMA period: {indicators.get('ema_period', 'N/A')}")
    try:
        new_ema = input("Enter new EMA period (or press Enter to keep current): ").strip()
        if new_ema:
            indicators["ema_period"] = int(new_ema)
    except ValueError:
        print("⚠️ Invalid period, keeping current value")
    
    print(f"Current CCI period: {indicators.get('cci_period', 'N/A')}")
    try:
        new_cci = input("Enter new CCI period (or press Enter to keep current): ").strip()
        if new_cci:
            indicators["cci_period"] = int(new_cci)
    except ValueError:
        print("⚠️ Invalid period, keeping current value")
    
    print(f"Current CCI tolerance: {indicators.get('cci_tolerance', 'N/A')}")
    try:
        new_tolerance = input("Enter new CCI tolerance (or press Enter to keep current): ").strip()
        if new_tolerance:
            indicators["cci_tolerance"] = int(new_tolerance)
    except ValueError:
        print("⚠️ Invalid tolerance, keeping current value")
    
    config["indicators"] = indicators
    if save_config(config):
        print("✅ Indicator settings saved")
    else:
        print("❌ Error saving settings")

def configure_system_settings(config: Dict[str, Any]):
    """Configure system settings"""
    print("\n🔧 System Settings:")
    
    system = config.get("system", {})
    
    print(f"Current log level: {system.get('log_level', 'N/A')}")
    new_log_level = input("Enter new log level (DEBUG/INFO/WARNING/ERROR) or press Enter to keep current: ").strip().upper()
    if new_log_level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
        system["log_level"] = new_log_level
    elif new_log_level:
        print("⚠️ Invalid log level, keeping current value")
    
    print(f"Current log to terminal: {system.get('log_to_terminal', 'N/A')}")
    new_terminal = input("Log to terminal? (true/false) or press Enter to keep current: ").strip().lower()
    if new_terminal in ["true", "false"]:
        system["log_to_terminal"] = new_terminal == "true"
    elif new_terminal:
        print("⚠️ Invalid value, keeping current value")
    
    config["system"] = system
    if save_config(config):
        print("✅ System settings saved")
    else:
        print("❌ Error saving settings")

def configure_user_settings(config: Dict[str, Any]):
    """Configure user settings"""
    print("\n👤 User Settings:")
    
    user = config.get("user", {})
    
    current_ssid = user.get("ssid", "")
    masked_ssid = current_ssid[:10] + "***" if len(current_ssid) > 10 else "Not set"
    print(f"Current SSID: {masked_ssid}")
    new_ssid = input("Enter new SSID (or press Enter to keep current): ").strip()
    if new_ssid:
        user["ssid"] = new_ssid
        print("✅ SSID updated")
    
    config["user"] = user
    if save_config(config):
        print("✅ User settings saved")
    else:
        print("❌ Error saving settings")

def view_full_config(config: Dict[str, Any]):
    """Display the full configuration"""
    print("\n📋 Full Configuration:")
    print("=" * 40)
    print(json.dumps(config, indent=2))
    input("\nPress Enter to continue...")

def join_community():
    """Open the Discord community link"""
    print("\n🌐 Opening Discord Community...")
    discord_url = "https://discord.gg/fDM43kPMwq"
    
    try:
        webbrowser.open(discord_url)
        print(f"✅ Opened Discord link: {discord_url}")
        print("🎉 Welcome to the RussBot community!")
        print("💬 Join us for discussions, updates, and support!")
    except Exception as e:
        print(f"❌ Could not open browser: {e}")
        print(f"🔗 Please manually open: {discord_url}")
    
    input("\nPress Enter to continue...")

def main():
    """Main execution function"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
    
    print("🤖 RussBot Trading Bot v2.0")
    print("🎯 Multi-Asset Trading System")
    print("=" * 40)
    
    while True:
        print("\n🚀 What would you like to do?")
        print("\n1. 🎯 Start Bot")
        print("2. ⚙️  Configure")
        print("3. 🌐 Join Community")
        print("4. 🚪 Exit")
        
        choice = input("\n👉 Select option (1-4): ").strip()
        
        if choice == "1":
            start_bot()
        elif choice == "2":
            configure_bot()
        elif choice == "3":
            join_community()
        elif choice == "4":
            print("\n👋 Thank you for using RussBot!")
            print("💰 Happy trading!")
            break
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
