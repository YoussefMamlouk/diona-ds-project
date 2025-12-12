Technical Report Template
=========================

Use this Markdown scaffold to write your final technical report. Convert to PDF using `pandoc` or LaTeX.

1. Abstract (≈200 words)
   - Short summary of research question, methods, and main findings.

2. Introduction
   - Motivation, context, and clear research question.

3. Related Work / Literature Review
   - Briefly situate your approach among related forecasting / structured-product research.

4. Data
   - Data sources and preprocessing steps.
   - Explain `--use-sample-data` fallback for grading.

5. Methodology
   - Model descriptions (ARIMA, AR(1), XGBoost, GARCH volatility), features, and hyperparameters.
   - Backtesting protocol and metrics (MAE, RMSE, MAPE).

6. Results
   - Present tables and figures comparing models by horizon and by metric.
   - Include representative forecasts and MC scenarios.

7. Discussion
   - Interpretation of results, limitations, robustness checks, and potential improvements.

8. Conclusion and Future Work

9. References
   - Cite libraries, data sources, and literature.

Appendix A: Code and Reproducibility
------------------------------------
- How to run: environment setup, tests, and example commands.

Build instructions (pandoc)
--------------------------
Install pandoc and a LaTeX engine (e.g., TexLive or MiKTeX). Then:

```bash
pandoc REPORT.md -o REPORT.pdf --from markdown --pdf-engine=xelatex --toc
```

Alternatively, use the provided LaTeX template (if you choose) and produce a PDF via `pdflatex`.
