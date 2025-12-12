# individual-project-Dionavdj
**Structured Product Recommender** This project analyzes historical market data to backtest and compare structured product strategies such as autocallables, reverse convertibles, and capital-protected notes. Based on risk tolerance and investment horizon, it recommends the strategy that historically achieved the best risk-adjusted return.

## Running the project

Recommended (clean, portable): install in editable mode and run the console script:

```powershell
python -m pip install -e .
# then run
forecast --demo --save
```

Developer (quick): run the package with `-m` or the provided launcher:

```powershell
# module mode (recommended for development)
python -m src --demo --save

# or the convenience launcher
python run.py --demo --save
```

Notes:

**Entrypoint (grading requirement)**

- The course requires that `python main.py` runs successfully. This repository provides a top-level `main.py` (root) that contains the full application and delegates to the package implementation. Use the top-level command in grading or when you want the exact entrypoint used by graders.

Example (run the deterministic demo and save output):

```powershell
python main.py --demo --use-sample-data --save
```

You can still run the package-style or the convenience launcher during development (`python -m src` or `python run.py`) — the top-level `main.py` is provided so `python main.py` works on a fresh environment as required by the course.

