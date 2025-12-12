"""Simple repo launcher for convenience.

Allows running the package without `-m` or installing the package:

    python run.py --demo --save

This calls `src.main:main` under package semantics.
"""
from src.main import main


if __name__ == "__main__":
    main()
