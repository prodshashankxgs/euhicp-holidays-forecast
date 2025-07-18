#!/usr/bin/env python3
"""
Setup script for EU HICP Package Holidays Price Forecast project.
Run this script to install dependencies and set up the project environment.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function."""
    print("🚀 EU HICP Package Holidays Forecast - Project Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("Consider using a virtual environment:")
        print("  python -m venv venv")
        print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Create necessary directories
    directories = ['data', 'models', 'outputs', 'figures']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        env_content = """# EU HICP Forecast API Configuration
# Get your FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_api_key_here

# Get your BLS API key from: https://www.bls.gov/developers/api_signature_v2.html
BLS_API_KEY=your_bls_api_key_here

# Optional settings
ECB_API_RATE_LIMIT=10
EUROSTAT_API_BASE_URL=https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file - please add your API keys")
    else:
        print("✅ .env file already exists")
    
    # Test imports
    print("\n📋 Testing imports...")
    test_imports = [
        ('polars', 'Polars data processing'),
        ('numpy', 'NumPy numerical computing'),
        ('plotly', 'Plotly visualization'),
        ('fredapi', 'FRED API client'),
        ('requests', 'HTTP requests'),
        ('statsmodels', 'Statistical models'),
        ('sklearn', 'Scikit-learn machine learning')
    ]
    
    failed_imports = []
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ❌ {description}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Some imports failed: {', '.join(failed_imports)}")
        print("Try reinstalling requirements: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test configuration
    print("\n🔧 Testing configuration...")
    try:
        from config import validate_api_keys
        validate_api_keys()
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        print("Please check your .env file and API keys")
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. 📝 Add your API keys to the .env file")
    print("2. 🚀 Start with: jupyter notebook 01_data_collection_and_cleaning.ipynb")
    print("3. 📊 Follow the phase-by-phase analysis")
    print("\nAPI Key Resources:")
    print("• FRED API: https://fred.stlouisfed.org/docs/api/api_key.html")
    print("• BLS API: https://www.bls.gov/developers/api_signature_v2.html")
    print("\nProject Structure:")
    print("• Phase 1: Data Collection & Cleaning")
    print("• Phase 2: Exploratory Data Analysis")
    print("• Phase 3: Seasonal Adjustment") 
    print("• Phase 4: Feature Engineering")
    print("• Phase 5: Model Development")
    print("• Phase 6: Forecasting & Validation")

if __name__ == "__main__":
    main() 