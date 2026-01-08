AI usage for this project
=========================

This file documents how AI tools were used during development, per course policy.

Tools used:
- OpenAI Codex (Codex CLI) for code edits, debugging support, and report drafting.
- ChatGPT for writing, clarification, and troubleshooting guidance.
- GitHub Copilot (assistant integrated in VS Code) for code suggestions and small helper functions.

Scope of AI assistance:
- Helped debug errors and clarify fixes during development.
- Suggested refactorings to modularize the pipeline across `src/data_loader.py`, `src/models.py`, `src/backtests.py`, `src/volatility.py`, `src/plots.py`, `src/results.py`, `src/evaluation.py`.
- Drafted helper functions and snippets that were reviewed and integrated by the author.
- Helped integrate cached Yahoo Finance loading and resampling logic.
- Assisted with generating the EDA summary outputs and plots.
- Suggested reproducibility improvements (fixed RNG seeds, single-thread settings, deterministic XGBoost configuration).
- Assisted with LaTeX report editing, including table/figure formatting and wording fixes.
- Helped update README documentation and project structure descriptions.

What was NOT done by AI:
- I validated, reviewed, and approved all code changes.
- I am responsible for the final report, interpretation of results, and model selection decisions.
