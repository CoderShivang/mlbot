#!/usr/bin/env python3
"""
Quick Setup Guide for ML Mean Reversion Bot
===========================================

Run this script to verify your setup and get started quickly.
"""

import os
import sys

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("✅ Python version OK")
    return True


def check_dependencies():
    """Check if dependencies are installed"""
    required = [
        'pandas',
        'numpy',
        'sklearn',
        'binance',
        'talib',
        'matplotlib'
    ]
    
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def verify_files():
    """Verify all required files exist"""
    required_files = [
        'ml_mean_reversion_bot.py',
        'example_usage.py',
        'live_trading_bot.py',
        'utils.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("\nVerifying files...")
    all_exist = True
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} - MISSING")
            all_exist = False
    
    return all_exist


def print_quick_start():
    """Print quick start guide"""
    print("\n" + "="*70)
    print("QUICK START GUIDE")
    print("="*70)
    
    print("\n📋 Step 1: Install Dependencies")
    print("-" * 70)
    print("pip install -r requirements.txt")
    print("\nNote: TA-Lib may require manual installation of C library first")
    print("See README.md for detailed instructions")
    
    print("\n🤖 Step 2: Train the Model")
    print("-" * 70)
    print("python example_usage.py")
    print("\nThis will:")
    print("  - Generate sample data (or fetch real data with API keys)")
    print("  - Train the ML model")
    print("  - Run backtest")
    print("  - Save trained model")
    
    print("\n📊 Step 3: Analyze Results")
    print("-" * 70)
    print("python utils.py")
    print("\nOr in Python:")
    print(">>> from utils import quick_analysis, create_report")
    print(">>> quick_analysis('trade_history.json')")
    print(">>> create_report('trade_history.json')")
    
    print("\n🚀 Step 4: Live Trading (TESTNET FIRST!)")
    print("-" * 70)
    print("1. Get Binance API keys from https://www.binance.com")
    print("2. Edit live_trading_bot.py and add your credentials")
    print("3. Set testnet=True for testing")
    print("4. Run: python live_trading_bot.py")
    
    print("\n⚠️  IMPORTANT WARNINGS")
    print("-" * 70)
    print("❗ Always test on Binance TESTNET first")
    print("❗ Never trade with money you can't afford to lose")
    print("❗ Start with small position sizes")
    print("❗ Monitor the bot regularly")
    print("❗ Use proper risk management")
    
    print("\n📚 Resources")
    print("-" * 70)
    print("📖 Full documentation: README.md")
    print("💻 Main bot code: ml_mean_reversion_bot.py")
    print("🧪 Example usage: example_usage.py")
    print("🔧 Utilities: utils.py")
    print("⚙️  Configuration: config_template.json")
    
    print("\n" + "="*70)
    print("Ready to start! Follow the steps above.")
    print("="*70 + "\n")


def main():
    """Main setup check"""
    print("\n" + "="*70)
    print("ML MEAN REVERSION BOT - SETUP CHECK")
    print("="*70 + "\n")
    
    # Check Python
    if not check_python_version():
        sys.exit(1)
    
    print("\nChecking dependencies...")
    print("-" * 70)
    deps_ok = check_dependencies()
    
    print("\nChecking files...")
    print("-" * 70)
    files_ok = verify_files()
    
    if deps_ok and files_ok:
        print("\n✅ All checks passed!")
        print_quick_start()
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies.")
        print("\nTo install dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
