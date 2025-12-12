# This module allows running the package with `python -m src`
# It imports from the root main.py
import sys
import pathlib

# Add parent directory to path to import root main
root_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from main import main

if __name__ == "__main__":
    main()
