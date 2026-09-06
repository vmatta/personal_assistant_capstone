"""Environment validation script - checks Python and pip versions before running."""

import sys
from packaging import version


def check_python_version():
    """Ensure Python 3.11.9 or higher is running."""
    min_version = "3.11.9"
    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if version.parse(current) < version.parse(min_version):
        print(f"❌ ERROR: Python {min_version}+ is required. You have {current}")
        print(f"   Install Python 3.11.9 from https://www.python.org/downloads/")
        sys.exit(1)
    
    print(f"✅ Python {current}")


def check_pip_version():
    """Ensure pip 26.1.1 or higher is installed."""
    import pip
    
    min_pip_version = "26.1.1"
    current_pip = pip.__version__
    
    if version.parse(current_pip) < version.parse(min_pip_version):
        print(f"❌ ERROR: pip {min_pip_version}+ is required. You have {current_pip}")
        print(f"   Upgrade pip with: python -m pip install --upgrade pip")
        sys.exit(1)
    
    print(f"✅ pip {current_pip}")


def validate_environment():
    """Run all environment checks."""
    print("\n🔍 Checking environment requirements...\n")
    
    try:
        check_python_version()
        check_pip_version()
        print("\n✅ All environment checks passed!\n")
    except SystemExit as e:
        if e.code != 0:
            print("\n❌ Environment validation failed. Please fix the issues above.\n")
            raise


if __name__ == "__main__":
    validate_environment()
