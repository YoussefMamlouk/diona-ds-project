#!/usr/bin/env python
"""
Script to verify that installed package versions match the pinned versions
in requirements.txt and environment.yml.

Run this script to ensure reproducibility across different devices.
"""
import sys
import importlib

# Expected versions from requirements.txt/environment.yml
EXPECTED_VERSIONS = {
    'numpy': '1.26.4',
    'pandas': '2.2.3',
    'matplotlib': '3.9.2',
    'scipy': '1.13.1',
    'yfinance': '0.2.25',
    'pmdarima': '2.1.1',
    'arch': '6.2.0',
    'xgboost': '2.1.4',
    'python-dotenv': '1.0.0',
    'requests': '2.31.0',
    'statsmodels': '0.14.6',
}

def get_package_version(package_name):
    """Get the installed version of a package."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            return module.__version__
        # Try alternative import names
        if package_name == 'python-dotenv':
            import dotenv
            return dotenv.__version__
        return None
    except ImportError:
        return None
    except Exception as e:
        print(f"Warning: Could not get version for {package_name}: {e}")
        return None

def main():
    """Check all package versions."""
    print("=" * 70)
    print("Package Version Verification")
    print("=" * 70)
    
    # Check Python version first
    python_version = sys.version_info
    expected_python_major = 3
    expected_python_minor = 10
    
    python_ok = (python_version.major == expected_python_major and 
                 python_version.minor == expected_python_minor)
    
    if python_ok:
        print(f"✓ Python version: {sys.version.split()[0]} (expected 3.10.x)")
    else:
        print(f"⚠️  Python version: {sys.version.split()[0]} (expected 3.10.x)")
        print(f"   For reproducibility, use Python 3.10 as specified in environment.yml")
    
    print()
    
    mismatches = []
    missing = []
    
    for package_name, expected_version in EXPECTED_VERSIONS.items():
        installed_version = get_package_version(package_name)
        
        if installed_version is None:
            missing.append(package_name)
            print(f"❌ {package_name:20s} - NOT INSTALLED (expected {expected_version})")
        elif installed_version != expected_version:
            mismatches.append((package_name, expected_version, installed_version))
            print(f"⚠️  {package_name:20s} - MISMATCH: expected {expected_version}, got {installed_version}")
        else:
            print(f"✓  {package_name:20s} - {installed_version}")
    
    print()
    print("=" * 70)
    
    if missing:
        print(f"❌ {len(missing)} package(s) not installed: {', '.join(missing)}")
    
    if mismatches:
        print(f"⚠️  {len(mismatches)} version mismatch(es) found:")
        for pkg, expected, installed in mismatches:
            print(f"   {pkg}: expected {expected}, got {installed}")
        print("\nTo fix, run:")
        print("   pip install -r requirements.txt")
        print("   or")
        print("   conda env update -f environment.yml")
        return 1
    
    if not missing and not mismatches and python_ok:
        print("✓ All package versions match expected versions!")
        print("✓ Python version is correct (3.10.x)")
        print("  Your environment is consistent and reproducible.")
        return 0
    elif not missing and not mismatches:
        print("✓ All package versions match expected versions!")
        print("⚠️  Python version mismatch - results may differ from conda environment")
        print("   To fix: Use Python 3.10 (conda env create -f environment.yml)")
        return 1
    
    return 1

if __name__ == "__main__":
    sys.exit(main())

