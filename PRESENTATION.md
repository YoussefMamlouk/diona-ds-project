Presentation Checklist & Script (8-10 minutes)
===============================================

Goal: Produce an 8–10 minute video that demonstrates the problem, approach, a live demo of `python main.py`, and key findings.

Suggested timing and content
----------------------------
- 0:00–1:30 — Motivation & Research Question
  - Explain why forecasting and structured-product selection matters.
- 1:30–3:30 — Technical Approach
  - Briefly describe data sources, features, and models (ARIMA, AR(1), GARCH, XGBoost).
  - Mention reproducibility (seeds, `--use-sample-data`).
- 3:30–7:00 — Live Demo (3.5 min)
  - Run: `python main.py --demo --use-sample-data --save`
  - Show printed backtest results, Monte Carlo percentiles, and the saved plot in `results/`.
  - Optionally run one live-data example (omit `--use-sample-data`) if internet is available.
- 7:00–8:30 — Results & Interpretation
  - Summarize which models worked best and why; show limitations.
- 8:30–9:30 — Conclusion & Future Work
  - Quick summary and next steps.

Recording tips
--------------
- Record terminal window and a small screen area showing the saved plot (crop if necessary).
- Run the demo ahead of time to ensure plots are produced and saved.
- Keep the terminal font readable and increase log verbosity only if needed.

Commands to prepare demo (copy-paste)
------------------------------------
```powershell
# create and activate environment (example using venv)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt

# demo (offline deterministic)
python main.py --demo --use-sample-data --save

# show saved plot
start results\forecast_TSLA_*.png
```

Notes
-----
- For the video, emphasize reproducibility and show the `PROPOSAL.md` and `REPORT.md` locations in the repo.
- State any limitations (data quality, short horizon forecasting difficulty) candidly — graders expect critical thinking.
