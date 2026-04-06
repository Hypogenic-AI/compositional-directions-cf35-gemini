# Which linear directions are compositional?

## Overview
This project investigates the compositionality of linear directions in Large Language Models (LLMs). Using Qwen-2.5-1.5B, we test the hypothesis that only related concepts compose naturally in the residual stream.

## Key Findings
- **The Compositionality Paradox:** Unrelated/novel concepts (e.g., "democracy car") are represented **more linearly** than common related concepts (e.g., "red car").
- **Stability:** Concept directions for abstract terms (Truth, Democracy) are 10-15% more consistent across contexts than color directions.
- **Steering:** Steering with "unrelated" directions is as effective as steering with "related" ones, challenging the notion of fixed conceptual subspaces.
- **Fusion:** Related concepts undergo "non-linear fusion" as they progress through the layers, while unrelated ones remain in linear superposition.

## How to Reproduce
1. **Environment:** Use `uv` to install dependencies from `pyproject.toml`.
2. **Activations:** Run `python3 src/extract_activations_v3.py` to extract hidden states.
3. **Analysis:** Run `python3 src/analyze_consistency.py` and `python3 src/systematic_steering.py`.

## Full Report
See [REPORT.md](REPORT.md) for detailed methodology, data construction, and theoretical interpretation.
