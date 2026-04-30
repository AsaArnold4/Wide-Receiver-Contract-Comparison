# WR Contract Comp Engine

**Live App → [appv9py-jtxcdvrf5cgnryh4k5asab.streamlit.app](https://appv9py-jtxcdvrf5cgnryh4k5asab.streamlit.app/)**

An interactive contract valuation tool for NFL wide receivers, built for sports agents preparing for contract negotiations. The app surfaces the most statistically similar historical signing events and projects a cap-adjusted market range for any WR in the PFF database — including players who have not yet signed a contract.

---

## What It Does

Search any wide receiver — signed or unsigned — and the engine returns:

- **Market Range** — lower, realistic, and upper bound APY projections based on the 15 most similar historical signings
- **Tier Representative Comps** — one specific signing event per tier that most closely mirrors the player's production profile, with full contract details
- **All Similar Signing Events** — ranked list of the 20 most comparable contracts with similarity scores, cap-adjusted APY, guaranteed money, and key stats
- **Agent Brief** — a plain-English paragraph summarizing the player's profile and market projection, ready to reference in a negotiation

---

## How the Similarity Score Works

1. **Feature engineering** — eight PFF route-running and efficiency metrics are computed as a weighted average across the player's three most recent seasons (0.6 / 0.3 / 0.1, most recent → oldest), matching the methodology used to build the contract pool
2. **LassoCV** — fits a regularized regression on 475 cap-adjusted signing events to identify which features actually predict contract value
3. **Permutation importance** — ranks each feature by its contribution to predictive accuracy, with a 5% floor to prevent any single feature from dominating
4. **Weighted cosine similarity** — each player's feature vector is scaled by importance weights before computing similarity against the full signing pool

Features used: YPRR, PFF route grade, receiving yards, TD rate, availability rate, average depth of target, target rate, drop rate.

---

## Data Sources

| Source | Usage |
|---|---|
| [Pro Football Focus (PFF)](https://www.pff.com/) | Per-season receiving stats (2013–2025) |
| [Over The Cap](https://overthecap.com/) | Contract values, cap-adjusted via annual cap inflation |

All contract values are inflated to the current salary cap year for apples-to-apples comparison across eras.

---

## Tech Stack

- **Python** — Streamlit, pandas, NumPy, scikit-learn
- **ML** — LassoCV (feature weighting), cosine similarity (player matching)
- **LLM** — Anthropic Claude API (agent brief generation)
- **Deployment** — Streamlit Community Cloud

---

## Repo Structure

```
├── app.py                        # Main Streamlit application
├── wr_comp_pool.csv              # 475 historical WR signing events with pre-signing PFF stats
├── wr_full_database.csv          # Extended WR contract database
├── 2013_receiving_summary.csv    # PFF season data (one file per year)
│   ...
└── 2025_receiving_summary.csv
```

---

Built by Asa Arnold · [GitHub](https://github.com/AsaArnold4/Wide-Receiver-Contract-Comparison)
