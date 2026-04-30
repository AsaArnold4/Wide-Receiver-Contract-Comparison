"""
WR Contract Similarity Engine — Streamlit App
Built by Asa Arnold for VaynerSports

Run with:  streamlit run app.py
Requires:  wr_comp_pool.csv, wr_full_database.csv, espn_headshot_ids.json,
           and all PFF season CSVs (e.g. 2024_receiving_summary.csv)
           in the same directory as this file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.inspection import permutation_importance
from numpy.linalg import norm
import re
import unicodedata
import glob
import os
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WR Contract Comp Engine",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080810;
    color: #ddddf0;
}
.stApp { background-color: #080810; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

.hero { padding: 2.5rem 0 0.5rem; }
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.6rem;
    letter-spacing: 0.05em;
    line-height: 1;
    color: #fff;
    margin: 0;
}
.hero-sub {
    font-size: 0.78rem;
    color: #55556a;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-top: 0.35rem;
}
.accent { color: #00e5a0; }

.stTextInput > div > div > input {
    background: #10101e !important;
    border: 1px solid #222238 !important;
    border-radius: 8px !important;
    color: #ddddf0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 2px rgba(0,229,160,0.12) !important;
}
.stTextInput > label {
    color: #7070a0 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}
.stNumberInput > div > div > input {
    background: #10101e !important;
    border: 1px solid #222238 !important;
    border-radius: 8px !important;
    color: #ddddf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stNumberInput > label {
    color: #7070a0 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
.stRadio > label { color: #7070a0 !important; font-size: 0.72rem !important; }

.stButton > button {
    background: #00e5a0 !important;
    color: #080810 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.65rem 1.8rem !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: #00ffb3 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,229,160,0.25) !important;
}

.card {
    background: #10101e;
    border: 1px solid #1c1c32;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.65rem;
}
.card-upper     { border-left: 3px solid #00e5a0; }
.card-realistic { border-left: 3px solid #4a8fe2; }
.card-lower     { border-left: 3px solid #e2604a; }

.range-wrap {
    background: #10101e;
    border: 1px solid #1c1c32;
    border-radius: 12px;
    padding: 1.6rem 2rem 1.4rem;
    margin-bottom: 1.4rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    gap: 1rem;
}
.range-cell { flex: 1; text-align: center; }
.range-label {
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.range-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.4rem;
    letter-spacing: 0.04em;
    line-height: 1;
}
.range-note { font-size: 0.68rem; color: #55556a; margin-top: 0.3rem; }
.arrow { flex: 0; color: #222238; font-size: 1.4rem; padding-bottom: 0.6rem; }

.player-header {
    background: #10101e;
    border: 1px solid #1c1c32;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0;
}

.chips { display:flex;gap:0.4rem;flex-wrap:wrap;margin-top:0.45rem; }
.chip {
    background: #18182c;
    border-radius: 6px;
    padding: 0.18rem 0.55rem;
    font-size: 0.7rem;
    color: #7070a0;
}
.chip b { color: #c0c0e0; font-weight: 500; }

.pill {
    font-size: 0.64rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.16rem 0.5rem;
    border-radius: 20px;
    font-weight: 700;
}
.pill-upper     { background:rgba(0,229,160,0.12); color:#00e5a0; }
.pill-realistic { background:rgba(74,143,226,0.12); color:#4a8fe2; }
.pill-lower     { background:rgba(226,96,74,0.12);  color:#e2604a; }

.sec-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.25rem;
    letter-spacing: 0.08em;
    color: #fff;
    margin: 1.6rem 0 0.8rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1c1c32;
}

.prof-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.7rem;
    margin-bottom: 1.4rem;
}
.prof-cell {
    background: #10101e;
    border: 1px solid #1c1c32;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.prof-val {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.75rem;
    letter-spacing: 0.04em;
    color: #00e5a0;
    line-height: 1;
}
.prof-lbl {
    font-size: 0.65rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #55556a;
    margin-top: 0.2rem;
}

.sim-badge {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.4rem;
    letter-spacing: 0.04em;
    padding: 0.25rem 0.65rem;
    border-radius: 7px;
    background: #18182c;
    white-space: nowrap;
    flex-shrink: 0;
}

/* Badge shown on player header for PFF-only players */
.pff-badge {
    display: inline-block;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.18rem 0.55rem;
    border-radius: 20px;
    font-weight: 700;
    background: rgba(255, 200, 50, 0.12);
    color: #ffc832;
    margin-left: 0.5rem;
    vertical-align: middle;
}

hr { border-color: #1c1c32 !important; margin: 1rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PFF SEASON DATA  —  load all per-season CSVs and build weighted profiles
# ══════════════════════════════════════════════════════════════════════════════

# Weighting scheme confirmed from comp_pool: 0.6 / 0.3 / 0.1 (most recent → oldest)
PFF_WEIGHTS = [0.6, 0.3, 0.1]

# Columns from PFF CSVs that map to the feature set
PFF_STAT_COLS = [
    "yprr", "grades_pass_route", "yards", "touchdowns",
    "routes", "targets", "player_game_count", "drop_rate",
    "avg_depth_of_target",
]

@st.cache_data(show_spinner="Loading PFF season data…")
def load_pff_seasons() -> dict[int, pd.DataFrame]:
    """Load all *_receiving_summary.csv files found next to app.py."""
    # Try multiple base directories to handle Streamlit Cloud path quirks
    candidates = [
        os.path.dirname(os.path.abspath(__file__)),   # next to script
        os.getcwd(),                                   # working directory
        os.path.join(os.getcwd(), "data"),             # data/ subfolder
    ]
    season_files = []
    for base in candidates:
        found = sorted(glob.glob(os.path.join(base, "*receiving_summary.csv")))
        if found:
            season_files = found
            break

    if not season_files:
        # Surface a clear error listing what IS in the script directory
        here = os.path.dirname(os.path.abspath(__file__))
        all_files = os.listdir(here)
        st.error(
            f"⚠️ No PFF season files found. "
            f"Files in app directory: {sorted(all_files)}"
        )
        st.stop()

    seasons = {}
    for path in season_files:
        try:
            year = int(re.search(r'(\d{4})', os.path.basename(path)).group(1))
            df = pd.read_csv(path)
            df["_season"] = year
            df["player_key"] = df["player"].apply(normalize_name)
            seasons[year] = df
        except Exception:
            pass
    return seasons


def build_pff_profile(player_name: str, pff_seasons: dict[int, pd.DataFrame],
                      as_of_year: int | None = None) -> dict | None:
    """
    Build a weighted stat profile for a player from PFF season CSVs.

    as_of_year: treat this as the year the contract would be signed —
                so we use data from (as_of_year-1) and earlier.
                Defaults to the most recent season available + 1.
    
    Returns a feature_dict matching FEATURE_COLS, or None if player not found.
    """
    key = normalize_name(player_name)
    available_years = sorted(pff_seasons.keys(), reverse=True)

    if not available_years:
        return None  # no PFF data loaded at all

    if as_of_year is None:
        as_of_year = max(available_years) + 1

    # Only use seasons prior to the signing year
    eligible_years = [y for y in available_years if y < as_of_year]

    # Collect up to 3 most-recent seasons where this player has data
    player_rows = []
    seasons_used = []
    for yr in eligible_years:
        df = pff_seasons[yr]
        match = df[df["player_key"] == key]
        if not match.empty:
            player_rows.append(match.iloc[0])
            seasons_used.append(yr)
        if len(player_rows) == 3:
            break

    if not player_rows:
        return None

    n = len(player_rows)
    # Trim weights to however many seasons we have, then re-normalise
    w = PFF_WEIGHTS[:n]
    w = [x / sum(w) for x in w]

    def wavg(col):
        vals = []
        for row in player_rows:
            v = row.get(col, np.nan)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(np.nan)
        # If any value is NaN, use only non-NaN ones with renormalised weights
        pairs = [(wt, v) for wt, v in zip(w, vals) if not np.isnan(v)]
        if not pairs:
            return np.nan
        total_w = sum(p[0] for p in pairs)
        return sum(p[0] * p[1] for p in pairs) / total_w

    # Determine season_max for most recent season (for availability_rate)
    most_recent_year = seasons_used[0]
    season_max = 17 if most_recent_year >= 2021 else 16

    w_games   = wavg("player_game_count")
    w_routes  = max(wavg("routes"), 1)
    w_targets = wavg("targets")
    w_tds     = wavg("touchdowns")

    feature_dict = {
        "yprr"               : wavg("yprr"),
        "grades_pass_route"  : wavg("grades_pass_route"),
        "yards"              : wavg("yards"),
        "td_rate"            : w_tds / w_routes,
        "availability_rate"  : min(w_games / season_max, 1.0),
        "avg_depth_of_target": wavg("avg_depth_of_target"),
        "target_rate"        : min(w_targets / w_routes, 1.0),
        "drop_rate"          : wavg("drop_rate"),
    }

    # Attach metadata for display
    feature_dict["_seasons_used"] = seasons_used
    feature_dict["_team"] = player_rows[0].get("team_name", "")
    feature_dict["_source"] = "pff"

    return feature_dict


def get_all_pff_players(pff_seasons: dict[int, pd.DataFrame]) -> list[str]:
    """Return all unique canonical player names across all PFF seasons."""
    names = set()
    for df in pff_seasons.values():
        names.update(df["player"].dropna().tolist())
    return sorted(names)


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE  —  mirrors wr_similarity_engine_v2.qmd exactly
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Building similarity engine…")
def load_engine():
    here      = os.path.dirname(os.path.abspath(__file__))
    comp_pool = pd.read_csv(os.path.join(here, "wr_comp_pool.csv"))
    full_db   = pd.read_csv(os.path.join(here, "wr_full_database.csv"))

    for df in [comp_pool, full_db]:
        df["signing_label"] = (
            df["player"] + " ("
            + df["year_signed"].astype(int).astype(str) + ", "
            + df["team"] + ")"
        )
        def season_max(y): return 17 if y >= 2021 else 16
        df["season_max"]        = df["year_signed"].apply(season_max)
        df["availability_rate"] = (df["player_game_count"] / df["season_max"]).clip(0, 1)
        df["target_rate"]       = (df["targets"] / df["routes"].clip(lower=1)).clip(0, 1)
        df["td_rate"]           = df["touchdowns"] / df["routes"].clip(lower=1)

    FEATURE_COLS = [
        "yprr", "grades_pass_route", "yards", "td_rate",
        "availability_rate", "avg_depth_of_target", "target_rate", "drop_rate",
    ]

    X = comp_pool[FEATURE_COLS].values
    y = comp_pool["inflated_apy"].values

    scaler_lasso = StandardScaler()
    X_std        = scaler_lasso.fit_transform(X)

    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, n_alphas=200)
    lasso_cv.fit(X_std, y)

    pi_result = permutation_importance(
        lasso_cv, X_std, y, n_repeats=50, random_state=42
    )
    pi_df = pd.DataFrame({
        "feature"   : FEATURE_COLS,
        "importance": pi_result.importances_mean,
    })
    pi_df["clipped"]    = pi_df["importance"].clip(lower=0)
    pi_df["pure"]       = pi_df["clipped"] / pi_df["clipped"].sum()
    FLOOR               = 0.05
    pi_df["hybrid_raw"] = pi_df["pure"].clip(lower=FLOOR)
    pi_df["weight"]     = pi_df["hybrid_raw"] / pi_df["hybrid_raw"].sum()

    WEIGHTS    = pi_df.set_index("feature")["weight"].to_dict()
    weight_vec = np.array([WEIGHTS[f] for f in FEATURE_COLS])
    W_sqrt     = np.sqrt(weight_vec)

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(comp_pool[FEATURE_COLS].values)
    X_weighted = X_scaled * W_sqrt

    return dict(
        comp_pool    = comp_pool,
        full_db      = full_db,
        FEATURE_COLS = FEATURE_COLS,
        WEIGHTS      = WEIGHTS,
        weight_vec   = weight_vec,
        W_sqrt       = W_sqrt,
        scaler       = scaler,
        X_weighted   = X_weighted,
    )


def normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = re.sub(r",?\s*jr\.?$", "", name)
    name = re.sub(r",?\s*sr\.?$", "", name)
    name = re.sub(r",?\s*ii+$",   "", name)
    name = name.replace(".", "").replace("'", "").replace("-", " ")
    return re.sub(r"\s+", " ", name).strip()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    d = norm(a) * norm(b)
    return float(np.dot(a, b) / d) if d != 0 else 0.0


def get_query_vector(feature_dict: dict, eng: dict) -> np.ndarray:
    raw    = np.array([feature_dict[f] for f in eng["FEATURE_COLS"]], dtype=float)
    scaled = eng["scaler"].transform(raw.reshape(1, -1))[0]
    return scaled * eng["W_sqrt"]


def find_similar_signings(
    qvec: np.ndarray, eng: dict, top_n: int = 20, exclude_key: str = None
) -> pd.DataFrame:
    cp     = eng["comp_pool"]
    scores = np.array([cosine_sim(qvec, eng["X_weighted"][i])
                       for i in range(len(eng["X_weighted"]))])
    results                     = cp.copy()
    results["similarity_score"] = scores
    results["guarantee_rate"]   = (
        results["guaranteed"] / results["total_value"]
    ).clip(0, 1)
    if exclude_key:
        results = results[results["player_key"] != exclude_key]
    results = results.sort_values("similarity_score", ascending=False).head(top_n)
    results["inflated_apy_m"]  = (results["inflated_apy"]       / 1e6).round(2)
    results["inflated_gtd_m"]  = (results["inflated_guaranteed"] / 1e6).round(2)
    results["apy_pct_cap_pct"] = (results["apy_pct_cap"]        * 100).round(2)
    return results.reset_index(drop=True)


def classify_tiers(results: pd.DataFrame):
    pool = results.head(15).copy()
    p25  = pool["inflated_apy"].quantile(0.25)
    p75  = pool["inflated_apy"].quantile(0.75)
    pool["tier"] = pool["inflated_apy"].apply(
        lambda v: "upper" if v >= p75 else ("lower" if v <= p25 else "realistic")
    )
    return pool, {"p25": p25, "p75": p75, "median": pool["inflated_apy"].median()}


def pick_reps(tiered: pd.DataFrame, pcts: dict) -> dict:
    def closest(df, target):
        if df.empty: return None
        return df.loc[(df["inflated_apy"] - target).abs().idxmin()]
    reps = {}
    for tier, key in [("upper","p75"), ("realistic","median"), ("lower","p25")]:
        sub        = tiered[tiered["tier"] == tier]
        rep        = closest(sub, pcts[key])
        reps[tier] = rep if rep is not None else closest(tiered, pcts[key])
    return reps


# ══════════════════════════════════════════════════════════════════════════════
#  PLAYER LOOKUP  —  comp pool first, PFF fallback
# ══════════════════════════════════════════════════════════════════════════════

def lookup_player(raw_name: str, eng: dict,
                  pff_seasons: dict[int, pd.DataFrame]) -> tuple[dict | None, str, str]:
    """
    Try to find a player's pre-signing production profile.

    Search order:
      1. wr_comp_pool (has both PFF stats + contract data)  → source = "pool"
      2. PFF season CSVs (stats only, no contract yet)      → source = "pff"

    Returns: (feature_dict, canonical_player_name, source)
    """
    key     = normalize_name(raw_name)
    cp      = eng["comp_pool"]
    matches = cp[cp["player_key"] == key]

    if not matches.empty:
        row          = matches.sort_values("year_signed", ascending=False).iloc[0]
        feature_dict = {f: row[f] for f in eng["FEATURE_COLS"]}
        feature_dict["_source"]       = "pool"
        feature_dict["_seasons_used"] = None
        feature_dict["_team"]         = row.get("team", "")
        return feature_dict, row["player"], key

    # Not in comp pool — try PFF CSVs
    pff_profile = build_pff_profile(raw_name, pff_seasons)
    if pff_profile is not None:
        # Find canonical name from PFF data
        for df in sorted(pff_seasons.keys(), reverse=True):
            match = pff_seasons[df][pff_seasons[df]["player_key"] == key]
            if not match.empty:
                canonical = match.iloc[0]["player"]
                return pff_profile, canonical, key

    return None, raw_name, key


def suggest_similar_names(raw_name: str, eng: dict,
                          pff_seasons: dict[int, pd.DataFrame]) -> list[str]:
    """Return fuzzy name suggestions from both comp pool and PFF data."""
    key = normalize_name(raw_name)[:4]
    suggestions = set()

    # From comp pool
    for p in eng["comp_pool"]["player"].unique():
        pk = normalize_name(p)
        if key in pk or pk[:4] == key[:4]:
            suggestions.add(p)

    # From PFF
    for df in pff_seasons.values():
        for p in df["player"].dropna().unique():
            pk = normalize_name(p)
            if key in pk or pk[:4] == key[:4]:
                suggestions.add(p)

    return sorted(suggestions)[:8]


# ══════════════════════════════════════════════════════════════════════════════
#  HEADSHOTS
# ══════════════════════════════════════════════════════════════════════════════




# ══════════════════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v: float) -> str:
    return f"${v/1e6:.1f}M"

def pill_html(tier: str) -> str:
    labels = {"upper":"Upper","realistic":"Realistic","lower":"Lower"}
    return f'<span class="pill pill-{tier}">{labels.get(tier, tier)}</span>'

def chip_html(label: str, val: str) -> str:
    return f'<div class="chip">{label}&nbsp;<b>{val}</b></div>'


def render_range_bar(pcts: dict):
    st.markdown(f"""
    <div class="range-wrap">
      <div class="range-cell">
        <div class="range-label" style="color:#e2604a;">Lower Bound</div>
        <div class="range-val"   style="color:#e2604a;">{fmt(pcts['p25'])}</div>
        <div class="range-note">25th pct · underpaid comps</div>
      </div>
      <div class="arrow">›</div>
      <div class="range-cell">
        <div class="range-label" style="color:#4a8fe2;">Realistic Market</div>
        <div class="range-val"   style="color:#4a8fe2;">{fmt(pcts['median'])}</div>
        <div class="range-note">median · market-rate comps</div>
      </div>
      <div class="arrow">›</div>
      <div class="range-cell">
        <div class="range-label" style="color:#00e5a0;">Upper Bound</div>
        <div class="range-val"   style="color:#00e5a0;">{fmt(pcts['p75'])}</div>
        <div class="range-note">75th pct · overpaid comps</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    if rep is None:
        return
    colors = {"upper":"#00e5a0", "realistic":"#4a8fe2", "lower":"#e2604a"}
    descs  = {
        "upper"    : "Paid above market for similar production — use as your ceiling ask",
        "realistic": "Paid at market rate for similar production — most likely outcome",
        "lower"    : "Paid below market for similar production — floor the agent must avoid",
    }
    c       = colors[tier]
    inj     = "⚠ injury year" if rep.get("injury_flag", 0) else ""
    gtd_pct = rep.get("guarantee_rate", 0) * 100
    apy_cap = rep["apy_pct_cap"] * 100

def render_tier_card(rep, tier: str):
    inj_html    = "&nbsp;" + '<span style="font-size:0.7rem;color:#c8924a;">' + inj + "</span>" if inj else ""
    signing     = rep["signing_label"]
    apy_str     = fmt(rep["inflated_apy"])
    gtd_str     = fmt(rep["inflated_guaranteed"])
    pff_str     = "{:.1f}".format(rep["grades_pass_route"])
    yprr_str    = "{:.2f}".format(rep["yprr"])
    yds_str     = "{:.0f}".format(rep["yards"])
    sim_str     = "{:.3f}".format(rep["similarity_score"])
    apy_cap_str = "{:.2f}%".format(apy_cap)
    gtd_pct_str = "{:.0f}%".format(gtd_pct)
    tier_upper  = tier.upper()
    desc        = descs[tier]

    chips = (
        chip_html("Cap-adj APY", apy_str) +
        chip_html("APY% of cap", apy_cap_str) +
        chip_html("Cap-adj Gtd", gtd_str) +
        chip_html("Gtd rate",    gtd_pct_str) +
        chip_html("PFF grade",   pff_str) +
        chip_html("YPRR",        yprr_str) +
        chip_html("Yards",       yds_str) +
        chip_html("Sim score",   sim_str)
    )

    html = (
        '<div class="card card-' + tier + '">' +
        '<div style="font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;' +
        'color:' + c + ';margin-bottom:0.55rem;font-weight:700;">' +
        tier_upper + " BOUND &nbsp;·&nbsp; " + desc +
        "</div>" +
        '<div style="display:flex;align-items:center;gap:1rem;">' +
        
        '<div style="flex:1;">' +
        '<div style="font-weight:700;font-size:1.05rem;color:#fff;margin-bottom:0.1rem;">' +
        signing + inj_html +
        "</div>" +
        '<div class="chips">' + chips + "</div>" +
        "</div></div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


    tier    = row["tier"]
    gtd_pct = row.get("guarantee_rate", 0) * 100
    inj     = "⚠" if row.get("injury_flag", 0) else ""
    sim     = row["similarity_score"]
    sim_col = "#00e5a0" if sim >= 0.95 else "#4a8fe2" if sim >= 0.90 else "#7070a0"
    apy_cap = row["apy_pct_cap_pct"]

    inj_html    = '<span style="color:#c8924a;font-size:0.72rem;">' + inj + "</span>" if inj else ""
    signing     = row["signing_label"]
    apy_str     = fmt(row["inflated_apy"])
    gtd_str     = fmt(row["inflated_guaranteed"])
    yprr_str    = "{:.2f}".format(row["yprr"])
    pff_str     = "{:.1f}".format(row["grades_pass_route"])
    yds_str     = "{:.0f}".format(row["yards"])
    adot_str    = "{:.1f}".format(row["avg_depth_of_target"])
    apy_cap_str = "{:.2f}%".format(apy_cap)
    gtd_pct_str = "{:.0f}%".format(gtd_pct)
    sim_str     = "{:.3f}".format(sim)
    pill        = pill_html(tier)

    chips = (
        chip_html("APY",   apy_str) +
        chip_html("Cap%",  apy_cap_str) +
        chip_html("Gtd",   gtd_str) +
        chip_html("Gtd%",  gtd_pct_str) +
        chip_html("YPRR",  yprr_str) +
        chip_html("PFF",   pff_str) +
        chip_html("Yards", yds_str) +
        chip_html("aDOT",  adot_str)
    )

    html = (
        '<div class="card" style="padding:0.9rem 1.2rem;">' +
        '<div style="display:flex;align-items:center;gap:0.9rem;">' +
        '<div style="font-family:Bebas Neue,sans-serif;font-size:1rem;' +
        'color:#2a2a44;width:1.4rem;text-align:right;flex-shrink:0;">' +
        str(rank) + "</div>" +
        
        '<div style="flex:1;min-width:0;">' +
        '<div style="display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;">' +
        '<span style="font-weight:600;font-size:0.95rem;color:#fff;">' + signing + "</span>" +
        inj_html + pill +
        "</div>" +
        '<div class="chips">' + chips + "</div>" +
        "</div>" +
        '<div class="sim-badge" style="color:' + sim_col + '">' + sim_str + "</div>" +
        "</div></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_profile_grid(fd: dict):
    cells = [
        ("YPRR",       f"{fd['yprr']:.2f}"),
        ("PFF Grade",  f"{fd['grades_pass_route']:.1f}"),
        ("Rec Yards",  f"{fd['yards']:.0f}"),
        ("Avail Rate", f"{fd['availability_rate']:.0%}"),
        ("Tgt Rate",   f"{fd['target_rate']:.3f}"),
        ("TD Rate",    f"{fd['td_rate']:.4f}"),
        ("aDOT",       f"{fd['avg_depth_of_target']:.1f}"),
        ("Drop Rate",  f"{fd['drop_rate']:.1f}%"),
    ]
    html = '<div class="prof-grid">'
    for label, val in cells:
        html += (
            f'<div class="prof-cell">'
            f'<div class="prof-val">{val}</div>'
            f'<div class="prof-lbl">{label}</div>'
            f'</div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    eng         = load_engine()
    pff_seasons = load_pff_seasons()
    cp          = eng["comp_pool"]

    # Header
    st.markdown("""
    <div class="hero">
      <div class="hero-title">WR CONTRACT <span class="accent">COMP</span> ENGINE</div>
      <div class="hero-sub">
        Cap-Adjusted Similarity &nbsp;·&nbsp; Built by Asa Arnold
        &nbsp;·&nbsp; VaynerSports Demo
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr/>", unsafe_allow_html=True)

    # Mode
    mode = st.radio(
        "mode",
        ["🔍  Search player in database", "✏️  Enter stats manually"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("<br/>", unsafe_allow_html=True)

    feature_dict = None
    player_name  = None
    player_key   = None
    run_query    = False

    # ── Mode 1: database lookup ────────────────────────────────────────────────
    if "Search" in mode:
        col_in, col_btn = st.columns([5, 1])
        with col_in:
            raw = st.text_input(
                "PLAYER NAME",
                placeholder="e.g. Tyreek Hill · Jordan Addison · Zay Flowers",
                key="search_input",
            )
        with col_btn:
            st.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
            go = st.button("FIND COMPS", key="go_btn")

        if raw and go:
            feature_dict, player_name, player_key = lookup_player(
                raw, eng, pff_seasons
            )

            if feature_dict is None:
                st.error(
                    f"**'{raw}'** not found in the contract pool or PFF database."
                )
                suggestions = suggest_similar_names(raw, eng, pff_seasons)
                if suggestions:
                    st.info(f"Did you mean: **{', '.join(suggestions)}**?")

            else:
                source = feature_dict.get("_source", "pool")
                run_query = True


                if source == "pool":
                    # Player has a contract record — show all contract years
                    matches  = cp[cp["player_key"] == player_key]
                    all_years = (
                        matches.sort_values("year_signed")["year_signed"]
                        .astype(int).tolist()
                    )
                    subtitle = (
                        f"Using pre-signing production from most recent contract record"
                        f"&nbsp;·&nbsp;"
                        f"Contracts in pool:&nbsp;{', '.join(str(y) for y in all_years)}"
                    )
                    badge = ""
                else:
                    # PFF-only player — unsigned / pre-contract
                    seasons_used = feature_dict.get("_seasons_used", [])
                    team         = feature_dict.get("_team", "")
                    subtitle = (
                        f"PFF data only — no contract on record yet"
                        f"&nbsp;·&nbsp;Seasons used:&nbsp;"
                        f"{', '.join(str(y) for y in seasons_used)}"
                        + (f"&nbsp;·&nbsp;{team}" if team else "")
                    )
                    badge = '<span class="pff-badge">PFF · Pre-Contract</span>'

                st.markdown(f"""
                <div class="player-header">
                      <div>
                    <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;
                                letter-spacing:0.05em;color:#fff;line-height:1;">
                      {player_name}{badge}
                    </div>
                    <div style="font-size:0.75rem;color:#55556a;margin-top:0.3rem;">
                      {subtitle}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Mode 2: manual stat entry ──────────────────────────────────────────────
    else:
        player_name = st.text_input(
            "PLAYER NAME (display only)",
            placeholder="e.g. Malik Nabers",
            key="manual_name",
        )
        st.markdown(
            "<div style='font-size:0.72rem;letter-spacing:0.1em;text-transform:uppercase;"
            "color:#7070a0;margin:0.75rem 0 0.5rem;'>"
            "Production stats — weighted across prior seasons</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            targets   = st.number_input("Targets",         value=130.0, step=1.0)
            routes    = st.number_input("Routes Run",       value=550.0, step=1.0)
            yards     = st.number_input("Receiving Yards",  value=1100.0, step=10.0)
            tds       = st.number_input("Touchdowns",       value=7.0,   step=1.0)
        with c2:
            yprr      = st.number_input("YPRR",            value=2.0,  step=0.01, format="%.2f")
            pff       = st.number_input("PFF Route Grade", value=80.0, step=0.1,  format="%.1f")
            adot      = st.number_input("aDOT",            value=10.0, step=0.1,  format="%.1f")
            drop      = st.number_input("Drop Rate (%)",   value=4.5,  step=0.1,  format="%.1f")

        c3, c4 = st.columns(2)
        with c3:
            games     = st.number_input("Games Played",    value=16.0, step=1.0)
        with c4:
            season_yr = st.number_input("Season Year",     value=2024, step=1, format="%d")

        routes_safe  = max(routes, 1)
        season_games = 17 if season_yr >= 2021 else 16

        feature_dict = {
            "yprr"               : yprr,
            "grades_pass_route"  : pff,
            "yards"              : yards,
            "td_rate"            : tds / routes_safe,
            "availability_rate"  : min(games / season_games, 1.0),
            "avg_depth_of_target": adot,
            "target_rate"        : min(targets / routes_safe, 1.0),
            "drop_rate"          : drop,
        }
        player_key = None

        st.markdown("<br/>", unsafe_allow_html=True)
        run_query = st.button("FIND COMPS", key="manual_go")

        if run_query and player_name:
            st.markdown(f"""
            <div class="player-header" style="margin-top:1rem;">
              <div>
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2rem;
                            letter-spacing:0.05em;color:#fff;line-height:1;">
                  {player_name}
                </div>
                <div style="font-size:0.75rem;color:#55556a;margin-top:0.3rem;">
                  Manual stat entry &nbsp;·&nbsp; {season_yr} season
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────────────────
    if run_query and feature_dict is not None:

        # Strip internal metadata keys before passing to the engine
        clean_fd = {k: v for k, v in feature_dict.items() if not k.startswith("_")}

        qvec            = get_query_vector(clean_fd, eng)
        results         = find_similar_signings(qvec, eng, top_n=20,
                                                exclude_key=player_key)
        tiered, pcts    = classify_tiers(results)
        reps            = pick_reps(tiered, pcts)

        st.markdown("<hr/>", unsafe_allow_html=True)

        # Input profile
        st.markdown('<div class="sec-title">Input Production Profile</div>',
                    unsafe_allow_html=True)
        render_profile_grid(clean_fd)

        # Market range
        st.markdown('<div class="sec-title">Market Range</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Based on the top-15 most similar contract-signing events, "
            "all values cap-adjusted to the current salary cap."
        )
        render_range_bar(pcts)

        # Tier comps
        st.markdown('<div class="sec-title">Tier Representative Comps</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Each card is a specific signing event (Player · Year · Team) whose "
            "pre-signing production profile most closely matches the query. "
            "Upper = overpaid relative to production. "
            "Lower = underpaid relative to production."
        )
        for tier in ["upper", "realistic", "lower"]:
            render_tier_card(reps[tier], tier)

        # Full ranked list
        st.markdown('<div class="sec-title">All Similar Signing Events</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Sorted by similarity score. Use this list to select the specific "
            "comps you want to cite across the negotiation table."
        )
        for rank, (_, row) in enumerate(tiered.iterrows(), start=1):
            render_result_row(row, rank)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.68rem;color:#22223a;text-align:center;'>"
            "Weights: LassoCV + permutation importance on 475 signing events · "
            "Cap inflation: Over The Cap · Grades: PFF Premium · "
            "Built by Asa Arnold"
            "</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
