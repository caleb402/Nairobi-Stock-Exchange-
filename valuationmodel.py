import pandas as pd
import numpy as np
from faker import Faker
from scipy.stats import ttest_ind, mannwhitneyu, normaltest
from scipy.stats import kruskal
import seaborn as sns
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — DATA GENERATION & VALUATION FLAGGING
# ══════════════════════════════════════════════════════════════════════════════

# ── Simulation Parameters ─────────────────────────────────────────────────────
N_COMPANIES = 64
DAYS        = 30
np.random.seed(42)

fake = Faker()
Faker.seed(42)

# Sector-specific drift (mu) and volatility (sigma) for GBM simulation
# Higher sigma = more volatile sector (e.g. Energy vs Banking)
SECTOR_PARAMS = {
    "Banking":       {"mu": 0.0004, "sigma": 0.015},
    "Manufacturing": {"mu": 0.0005, "sigma": 0.02},
    "Telecom":       {"mu": 0.0006, "sigma": 0.018},
    "Energy":        {"mu": 0.0007, "sigma": 0.03},
    "Retail":        {"mu": 0.0005, "sigma": 0.022},
    "Agriculture":   {"mu": 0.0003, "sigma": 0.025}
}

# Sector-specific volume parameters for Amihud liquidity simulation
# mean       = average daily trading volume in KSh
# zero_prob  = probability of a zero-volume day (stock did not trade)
# Agriculture has the highest zero_prob — NSE agricultural stocks
# routinely go entire weeks without a single trade
SECTOR_VOLUME_PARAMS = {
    "Banking":       {"mean": 50e6,  "zero_prob": 0.02},
    "Manufacturing": {"mean": 10e6,  "zero_prob": 0.10},
    "Telecom":       {"mean": 200e6, "zero_prob": 0.01},
    "Energy":        {"mean": 5e6,   "zero_prob": 0.20},
    "Retail":        {"mean": 3e6,   "zero_prob": 0.25},
    "Agriculture":   {"mean": 1e6,   "zero_prob": 0.40},
}

AMIHUD_VETO_THRESHOLD = 1e-5          # stocks above this are illiquid traps
VETO_DOWNGRADE_FLAG   = "Weakly Undervalued"   # what illiquid stocks become

def generate_companies() -> pd.DataFrame:
    """
    Generate a synthetic NSE company universe with fundamental metrics.

    Returns
    -------
    pd.DataFrame with Ticker, Company Name, Sector, and financial ratios.
    """
    tickers      = [f"COMP{i:02d}" for i in range(1, N_COMPANIES + 1)]
    names        = [fake.company() for _ in range(N_COMPANIES)]
    sectors      = np.random.choice(list(SECTOR_PARAMS.keys()), N_COMPANIES)
    market_caps  = np.random.uniform(1,   200, N_COMPANIES).round(2)
    pe_ratios    = np.random.uniform(5,    25, N_COMPANIES).round(2)
    pb_ratios    = np.random.uniform(0.5,   5, N_COMPANIES).round(2)
    div_yields   = np.random.uniform(0,    10, N_COMPANIES).round(2)
    roes         = np.random.uniform(5,    30, N_COMPANIES).round(2)

    return pd.DataFrame({
        "Ticker":              tickers,
        "Company Name":        names,
        "Sector":              sectors,
        "Market Cap (KES B)":  market_caps,
        "P/E Ratio":           pe_ratios,
        "P/B Ratio":           pb_ratios,
        "Dividend Yield (%)":  div_yields,
        "ROE (%)":             roes
    })


# ══════════════════════════════════════════════════════════════════════════════
# SECTOR-SPECIFIC CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Criterion weights per sector ──────────────────────────────────────────────
#
# Three core criteria, each weighted differently per sector based on which
# fundamental metrics carry the most economic signal in that industry:
#
# roe_pb  (ROE-to-PB Efficiency):
#   Measures earnings quality per unit of book value.
#   Most meaningful where book value reflects real asset worth (Banking,
#   Manufacturing). Less meaningful where assets are intangible (Telecom)
#   or economically depreciated differently from accounting (Energy).
#
# roe_pe  (ROE-to-PE Efficiency):
#   Measures earnings quality per unit of earnings multiple.
#   Most meaningful where earnings are stable and predictable (Telecom,
#   Energy). Less meaningful where earnings are seasonal or cyclical
#   (Agriculture) or distorted by provisioning (Banking).
#
# simple  (Simple Threshold — Graham-style screen):
#   Requires P/E < avg AND P/B < avg AND ROE > avg simultaneously.
#   Most meaningful where both multiples are reliable (Retail, Manufacturing).
#   Weakened wherever either P/E or P/B is unreliable as a valuation metric.
#
# Weight rationale per sector:
# ────────────────────────────────────────────────────────────────────────────
# Banking      : P/B is the primary bank valuation metric globally (loans are
#                financial assets marked to market). P/E distorted by loan
#                provisioning decisions. ROE/PB weighted highest.
#
# Manufacturing: Both P/E and P/B reliable for tangible-asset businesses.
#                Balanced weights with slight ROE/PB preference.
#
# Telecom      : Asset-light model — spectrum/brand = intangible assets.
#                P/B unreliable. P/E drives valuation. ROE/PE weighted highest.
#
# Energy       : Heavy capex, long-life physical assets depreciate on
#                accounting schedules disconnected from economic reality.
#                P/B unreliable. P/E more relevant. Simple threshold weakened.
#
# Retail       : Thin margins make ROE volatile. Inventory/store assets
#                depreciate fast making P/B unreliable. Simple threshold
#                (requiring ALL three conditions) is the most robust screen.
#
# Agriculture  : Highly seasonal earnings make P/E volatile and unreliable.
#                Land and physical assets (P/B) more stable. ROE/PE
#                given very low weight due to earnings cyclicality.
# ────────────────────────────────────────────────────────────────────────────
SECTOR_CRITERION_WEIGHTS: dict[str, dict[str, float]] = {
    "Banking": {
        "roe_pb": 3.5,   # PRIMARY — book value is meaningful for banks
        "roe_pe": 1.5,   # SECONDARY — P/E distorted by provisioning
        "simple": 2.0,   # SUPPORTING — useful but not lead indicator
    },
    "Manufacturing": {
        "roe_pb": 3.0,   # PRIMARY — tangible assets make P/B reliable
        "roe_pe": 2.5,   # SECONDARY — stable earnings make P/E reliable
        "simple": 2.0,   # SUPPORTING — balanced screen
    },
    "Telecom": {
        "roe_pb": 1.5,   # WEAK — asset-light, P/B less meaningful
        "roe_pe": 3.5,   # PRIMARY — earnings-driven valuation dominates
        "simple": 2.0,   # SUPPORTING — still useful as confirmation
    },
    "Energy": {
        "roe_pb": 1.5,   # WEAK — physical asset book values unreliable
        "roe_pe": 3.5,   # PRIMARY — earnings multiple more relevant
        "simple": 1.5,   # WEAK — P/B component undermines this screen
    },
    "Retail": {
        "roe_pb": 1.5,   # WEAK — inventory/fit-out assets depreciate fast
        "roe_pe": 2.0,   # MODERATE — margins volatile but P/E still useful
        "simple": 3.5,   # PRIMARY — all-three-must-hold is strongest screen
    },
    "Agriculture": {
        "roe_pb": 3.0,   # PRIMARY — land/asset values more stable than earnings
        "roe_pe": 1.0,   # WEAK — seasonal earnings make P/E unreliable
        "simple": 2.0,   # SUPPORTING — P/E component weakens it but useful
    },
}

# Fallback weights for any sector not in the lookup table
DEFAULT_CRITERION_WEIGHTS: dict[str, float] = {
    "roe_pb": 3.0,
    "roe_pe": 2.5,
    "simple": 2.0,
}

# ── Dividend yield weights per sector ────────────────────────────────────────
#
# Reflects how reliably dividend yield signals genuine value in each sector.
# 0.0 = excluded from scoring entirely for that sector.
#
# Banking      : Mature, regulated, cash-generative. Dividends are primary
#                shareholder return mechanism on NSE (KCB, Equity, Co-op).
#                Strongest dividend signal. Weight: 1.5
#
# Telecom      : Safaricom historically strong dividend payer. However,
#                continuous capex for infrastructure reinvestment means
#                yield must be interpreted carefully. Weight: 1.0
#
# Retail       : Consistent cash flow generators reward via dividends but
#                Kenyan retail margins are thin and payments can be irregular.
#                Weight: 0.8
#
# Manufacturing: Varies widely — large established manufacturers (EABL, BAT)
#                pay reliably, smaller ones erratically. Low weight. Weight: 0.5
#
# Agriculture  : Seasonal and cyclical earnings produce inconsistent dividends.
#                One-off special dividends or collapsed share prices can
#                inflate yield misleadingly. Very low weight. Weight: 0.2
#
# Energy       : Heavy capex requirement means high yield often signals
#                underinvestment in growth (KenGen dynamic). High yield
#                is a RED FLAG not a value signal. Excluded entirely. Weight: 0.0
# ────────────────────────────────────────────────────────────────────────────
DIVIDEND_WEIGHTS: dict[str, float] = {
    "Banking":       1.5,
    "Telecom":       1.0,
    "Retail":        0.8,
    "Manufacturing": 0.5,
    "Agriculture":   0.2,
    "Energy":        0.0,   # excluded — high yield = capex underinvestment risk
}

# ── Payout ratio sustainability ceiling ──────────────────────────────────────
# Companies paying out more than 85% of earnings are flagged as unsustainable.
# A single bad quarter could force a dividend cut, destroying shareholder value.
# The guard prevents distressed companies with collapsing share prices
# (which mechanically inflate yield) from scoring highly on dividend yield.
#
# Formula used:  Payout Ratio ≈ Dividend Yield (%) × P/E Ratio / 100
# Derivation:
#   Yield   = Dividend per Share / Share Price
#   P/E     = Share Price / Earnings per Share
#   Yield × P/E = Dividend / Earnings = Payout Ratio
# ────────────────────────────────────────────────────────────────────────────
MAX_PAYOUT_RATIO: float = 0.85

# ── Flag thresholds on normalised 0.0–1.0 scale ──────────────────────────────
FLAG_THRESHOLDS: dict[str, float] = {
    "Strongly Undervalued":   0.75,
    "Moderately Undervalued": 0.50,
    "Weakly Undervalued":     0.25,
    # Below 0.25 → Overvalued
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def is_dividend_sustainable(div_yield: float, pe_ratio: float) -> bool:
    """
    Determine whether a company's dividend yield is financially sustainable.

    Uses the approximation:
        Payout Ratio ≈ Dividend Yield (%) × P/E Ratio / 100

    Returns False (unsustainable) if:
      - P/E ratio is missing, zero, or negative (can't compute)
      - Approximated payout ratio exceeds MAX_PAYOUT_RATIO (85%)

    Parameters
    ----------
    div_yield : Dividend yield as a percentage (e.g. 5.0 for 5%)
    pe_ratio  : Price-to-Earnings ratio

    Returns
    -------
    bool — True if dividend appears sustainable, False otherwise

    Examples
    --------
    >>> is_dividend_sustainable(10.0, 5.0)
    True   # payout ≈ 50% — sustainable
    >>> is_dividend_sustainable(10.0, 15.0)
    False  # payout ≈ 150% — unsustainable
    """
    if pe_ratio is None or pe_ratio <= 0:
        return False
    approx_payout = (div_yield * pe_ratio) / 100
    return approx_payout <= MAX_PAYOUT_RATIO


def score_dividend_yield(
    company_yield:    float,
    sector_avg_yield: float,
    sector:           str,
    pe_ratio:         float
) -> float:
    """
    Compute the weighted dividend yield score contribution for one company.

    Scoring Rules (applied in order):
    ──────────────────────────────────────────────────────────────────────
    1. If sector dividend weight is 0.0 (e.g. Energy) → return 0.0
       Rationale: high yield in Energy signals capex underinvestment

    2. If payout ratio sustainability check fails → return 0.0
       Rationale: unsustainable dividends will be cut; rewarding them
       would incorrectly flag distressed companies as undervalued

    3. If company yield > sector average yield → return full sector weight
       Rationale: above-average yield with sustainable payout = genuine
       income value signal relative to sector peers

    4. If company yield > 0 but <= sector average → return half sector weight
       Rationale: pays a dividend (positive signal) but not exceptional

    5. No dividend paid → return 0.0

    Parameters
    ----------
    company_yield    : company's dividend yield (%)
    sector_avg_yield : mean dividend yield for the sector (%)
    sector           : sector name string
    pe_ratio         : company's P/E ratio (for payout ratio approximation)

    Returns
    -------
    float — weighted score contribution (0.0 to sector dividend weight)
    """
    weight = DIVIDEND_WEIGHTS.get(sector, 0.0)

    # Rule 1: Sector excluded from dividend scoring
    if weight == 0.0:
        return 0.0

    # Rule 2: Payout ratio guard — blocks unsustainable yields
    if not is_dividend_sustainable(company_yield, pe_ratio):
        return 0.0

    # Rule 3: Above-average sustainable yield — full credit
    if company_yield > sector_avg_yield:
        return weight

    # Rule 4: Positive but below-average yield — partial credit
    if company_yield > 0:
        return weight * 0.5

    # Rule 5: No dividend
    return 0.0


def get_sector_weights(sector: str) -> dict[str, float]:
    """
    Retrieve criterion weights for a given sector.
    Falls back to DEFAULT_CRITERION_WEIGHTS for unrecognised sectors,
    ensuring the function never raises a KeyError on new sector names.

    Parameters
    ----------
    sector : sector name string

    Returns
    -------
    dict with keys: roe_pb, roe_pe, simple
    """
    return SECTOR_CRITERION_WEIGHTS.get(sector, DEFAULT_CRITERION_WEIGHTS)


def compute_sector_max(sector: str) -> float:
    """
    Compute the maximum possible raw score for a given sector.

    This is the sum of all criterion weights (core + dividend) assuming
    a hypothetical company that passes every single criterion perfectly.
    Used as the denominator in score normalisation to produce a 0.0–1.0
    scale that is comparable across sectors with different weight totals.

    Parameters
    ----------
    sector : sector name string

    Returns
    -------
    float — maximum achievable raw score for this sector
    """
    cw = get_sector_weights(sector)
    return cw["roe_pb"] + cw["roe_pe"] + cw["simple"] + DIVIDEND_WEIGHTS.get(sector, 0.0)


def assign_flag_from_normalised_score(score: float,thresholds: dict = None ) -> str:
    """
    Map a normalised score (0.0–1.0) to a valuation flag string.

    Thresholds (from FLAG_THRESHOLDS constant):
        >= 0.75 → Strongly Undervalued
        >= 0.50 → Moderately Undervalued
        >= 0.25 → Weakly Undervalued
        <  0.25 → Overvalued

    Parameters
    ----------
    score : normalised valuation score between 0.0 and 1.0

    Returns
    -------
    str — valuation flag label
    """
    t = thresholds if thresholds is not None else FLAG_THRESHOLDS

    if score >= t["Strongly Undervalued"]:
        return "Strongly Undervalued"
    if score >= t["Moderately Undervalued"]:
        return "Moderately Undervalued"
    if score >= t["Weakly Undervalued"]:
        return "Weakly Undervalued"
    return "Overvalued"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def apply_valuation_flags(companies: pd.DataFrame,thresholds: dict = None) -> pd.DataFrame:
    """
    Score each NSE company using four sector-specific weighted criteria
    and assign a comprehensive valuation flag.

    ── Four Scoring Criteria ────────────────────────────────────────────────

    Criterion 1 — ROE-to-PB Efficiency  (weight: sector-specific, max 3.5)
        Condition: P/E < sector avg
                   AND ROE > sector avg
                   AND (ROE / P/B) > (sector_avg_ROE / sector_avg_PB)
        Signal: Company generates better earnings per unit of book value
                than its sector peers — strongest signal in Banking and
                Manufacturing where book value reflects real asset worth.

    Criterion 2 — ROE-to-PE Efficiency  (weight: sector-specific, max 3.5)
        Condition: P/B < sector avg
                   AND ROE > sector avg
                   AND (ROE / P/E) > (sector_avg_ROE / sector_avg_PE)
        Signal: Company generates better earnings per unit of earnings
                multiple than peers — strongest in Telecom and Energy
                where earnings stability makes P/E the primary metric.

    Criterion 3 — Simple Threshold      (weight: sector-specific, max 3.5)
        Condition: P/E < sector avg
                   AND P/B < sector avg
                   AND ROE > sector avg
        Signal: Classic Graham-style screen — cheap on both multiples
                AND operationally efficient. Strongest in Retail where
                requiring all three simultaneously filters noise best.

    Criterion 4 — Dividend Yield        (weight: sector-specific, 0.0–1.5)
        Condition: Yield > sector avg yield
                   AND payout ratio <= 85% (sustainability guard)
        Signal: Company shares sustainable profits with shareholders —
                most meaningful in Banking (1.5), excluded in Energy (0.0).
        Partial credit (50%) for positive yield below sector average.

    ── Score Normalisation ──────────────────────────────────────────────────

    Raw Score   = sum of weighted criteria passed
    Sector Max  = sum of ALL weights for that sector (theoretical maximum)
    Normalised  = Raw Score / Sector Max  → range: 0.0 to 1.0

    Normalisation ensures a Banking score of 0.82 is directly comparable
    to a Telecom score of 0.82 — both represent the same relative strength
    of undervaluation within their sector, despite having different raw
    score maximums due to different dividend weights.

    ── Flag Thresholds (normalised 0.0–1.0 scale) ───────────────────────────

        >= 0.75 → Strongly Undervalued
        >= 0.50 → Moderately Undervalued
        >= 0.25 → Weakly Undervalued
        <  0.25 → Overvalued

    ── Key Differences vs Equal-Weight Approach ─────────────────────────────

        Equal-weight: ROE/PB = ROE/PE = Simple = 1 point each
        This code:    ROE/PB = 3.5 in Banking vs 1.5 in Telecom
                      ROE/PE = 3.5 in Telecom vs 1.0 in Agriculture
                      Simple = 3.5 in Retail  vs 1.5 in Energy
                      Dividend: 1.5 in Banking, 0.0 in Energy (excluded)

        Result: Same company fundamentals produce different scores across
        sectors — correctly reflecting that P/B means more for a bank
        than for a telecom operator.

    Parameters
    ----------
    companies : pd.DataFrame containing at minimum:
                  Sector, P/E Ratio, P/B Ratio, ROE (%), Dividend Yield (%)

    Returns
    -------
    pd.DataFrame with four new columns added:
        Raw Valuation Score         : weighted sum of criteria passed
        Normalised Valuation Score  : raw score / sector max (0.0–1.0)
        Valuation Flag Comprehensive: Strongly/Moderately/Weakly Undervalued
                                      or Overvalued
        Score Breakdown             : human-readable string showing per-criterion
                                      contribution for auditability
    """
    # ── Input validation ──────────────────────────────────────────────────────
    required_cols = {
        "Sector", "P/E Ratio", "P/B Ratio", "ROE (%)", "Dividend Yield (%)"
    }
    missing = required_cols - set(companies.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if companies.empty:
        raise ValueError("Input DataFrame is empty.")

    companies = companies.copy()
    companies["Raw Valuation Score"]          = 0.0
    companies["Normalised Valuation Score"]   = 0.0
    companies["Valuation Flag Comprehensive"] = ""
    companies["Score Breakdown"]              = ""

    active_thresholds = thresholds if thresholds is not None else FLAG_THRESHOLDS

    for sector in companies["Sector"].unique():
        sec_mask = companies["Sector"] == sector
        sec_data = companies[sec_mask]

        # ── Sector averages ───────────────────────────────────────────────────
        avg_pe    = sec_data["P/E Ratio"].mean()
        avg_pb    = sec_data["P/B Ratio"].mean()
        avg_roe   = sec_data["ROE (%)"].mean()
        avg_yield = sec_data["Dividend Yield (%)"].mean()

        # ── Sector-specific weights ───────────────────────────────────────────
        cw         = get_sector_weights(sector)
        sector_max = compute_sector_max(sector)

        # ── Criterion 1: ROE-to-PB Efficiency ────────────────────────────────
        # Earnings quality per unit of book value vs sector peers
        mask_roe_pb = (
            sec_mask &
            (companies["P/E Ratio"] < avg_pe) &
            (companies["ROE (%)"]   > avg_roe) &
            (
                (companies["ROE (%)"] / companies["P/B Ratio"])
                > (avg_roe / avg_pb)
            )
        )

        # ── Criterion 2: ROE-to-PE Efficiency ────────────────────────────────
        # Earnings quality per unit of earnings multiple vs sector peers
        mask_roe_pe = (
            sec_mask &
            (companies["P/B Ratio"] < avg_pb) &
            (companies["ROE (%)"]   > avg_roe) &
            (
                (companies["ROE (%)"] / companies["P/E Ratio"])
                > (avg_roe / avg_pe)
            )
        )

        # ── Criterion 3: Simple Threshold ─────────────────────────────────────
        # Classic Graham screen — cheap on both multiples AND efficient
        mask_simple = (
            sec_mask &
            (companies["P/E Ratio"] < avg_pe) &
            (companies["P/B Ratio"] < avg_pb) &
            (companies["ROE (%)"]   > avg_roe)
        )

        # ── Criterion 4: Dividend Yield (row-wise, sector-specific weight) ────
        # Payout ratio guard applied inside score_dividend_yield()
        div_scores = companies[sec_mask].apply(
            lambda row: score_dividend_yield(
                row["Dividend Yield (%)"],
                avg_yield,
                sector,
                row["P/E Ratio"]
            ),
            axis=1
        )

        # ── Weighted composite raw score ──────────────────────────────────────
        c1_scores = mask_roe_pb.astype(float) * cw["roe_pb"]
        c2_scores = mask_roe_pe.astype(float) * cw["roe_pe"]
        c3_scores = mask_simple.astype(float) * cw["simple"]
        c4_scores = div_scores.reindex(companies.index, fill_value=0.0)

        raw_scores = c1_scores + c2_scores + c3_scores + c4_scores
        companies.loc[sec_mask, "Raw Valuation Score"] = raw_scores

        # ── Normalised score (0.0–1.0) ────────────────────────────────────────
        # Dividing by sector_max makes scores cross-sector comparable
        # despite different sectors having different weight totals
        normalised = raw_scores / sector_max
        companies.loc[sec_mask, "Normalised Valuation Score"] = normalised.round(4)

         
        # ── REPLACEMENT — uses active_thresholds which can be calibrated or default ──
        companies.loc[sec_mask, "Valuation Flag Comprehensive"] = (
            normalised.apply(
        lambda s: assign_flag_from_normalised_score(s, active_thresholds)
              )
       )

        # ── Score breakdown string (for auditability and reporting) ───────────
        # Shows exactly which criteria each company passed and at what weight
        sector_indices = companies[sec_mask].index

        for idx in sector_indices:
            c1 = c1_scores.loc[idx] if idx in c1_scores.index else 0.0
            c2 = c2_scores.loc[idx] if idx in c2_scores.index else 0.0
            c3 = c3_scores.loc[idx] if idx in c3_scores.index else 0.0
            c4 = c4_scores.loc[idx] if idx in c4_scores.index else 0.0

            breakdown = (
                f"ROE/PB={c1:.1f}({cw['roe_pb']}) | "
                f"ROE/PE={c2:.1f}({cw['roe_pe']}) | "
                f"Simple={c3:.1f}({cw['simple']}) | "
                f"Div={c4:.2f}({DIVIDEND_WEIGHTS.get(sector, 0.0)}) | "
                f"Raw={c1+c2+c3+c4:.2f}/{sector_max:.1f} | "
                f"Norm={normalised.loc[idx]:.4f}"
            )
            companies.loc[idx, "Score Breakdown"] = breakdown

    return companies


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC UTILITY
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC UTILITY                          ← KEEP THIS HEADER
# ══════════════════════════════════════════════════════════════════════════════
def calibrate_thresholds(companies: pd.DataFrame) -> dict:
    """
    Suggest thresholds based on percentiles of the actual
    normalised score distribution.
    Uses only non-zero scores for percentile calculation to
    preserve the Overvalued tier for genuine zero-scorers.
    Ensures minimum spacing between thresholds and that the
    Weakly Undervalued floor sits above the minimum non-zero
    score so companies scoring at or near 0.0 remain Overvalued.
    """
    scores = companies["Normalised Valuation Score"]

    zero_pct = (scores == 0).mean() * 100
    print(f"\nScore distribution summary:")
    print(f"  Companies scoring 0.0:     {zero_pct:.1f}%")
    print(f"  Companies scoring above 0: {100 - zero_pct:.1f}%")

    nonzero_scores = scores[scores > 0]

    if len(nonzero_scores) < 4:
        print("\n[WARNING] Too few non-zero scores for calibration.")
        return {
            "Strongly Undervalued":   0.60,
            "Moderately Undervalued": 0.40,
            "Weakly Undervalued":     0.20,
        }

    # ── Use only non-zero scores for ALL percentiles ──────────────────────────
    # This preserves the Overvalued tier for genuine zero-scorers
    p75 = round(nonzero_scores.quantile(0.75), 4)
    p50 = round(nonzero_scores.quantile(0.50), 4)
    p25 = round(nonzero_scores.quantile(0.25), 4)

    # ── Ensure minimum spacing between thresholds ─────────────────────────────
    MIN_SPACING = 0.05
    if p75 - p50 < MIN_SPACING:
        p75 = round(p50 + MIN_SPACING, 4)
    if p50 - p25 < MIN_SPACING:
        p50 = round(p25 + MIN_SPACING, 4)
    if p75 <= p50:
        p75 = round(p50 + MIN_SPACING, 4)

    # ── Critical: Weakly Undervalued must be above the minimum non-zero score──
    # This guarantees companies scoring at or near 0.0 remain Overvalued
    min_nonzero = round(nonzero_scores.min(), 4)
    if p25 <= min_nonzero:
        p25 = round(min_nonzero + 0.001, 4)
        p50 = max(p50, round(p25 + MIN_SPACING, 4))
        p75 = max(p75, round(p50 + MIN_SPACING, 4))

    suggested = {
        "Strongly Undervalued":   p75,
        "Moderately Undervalued": p50,
        "Weakly Undervalued":     p25,
    }

    print("\nSuggested thresholds based on non-zero score distribution:")
    for flag, threshold in suggested.items():
         print(f"  >= {threshold:.4f} → {flag}")
    print(f"  <  {p25:.4f} → Overvalued")

    return suggested

def print_weight_summary() -> None:
    """
    Print a readable summary of all sector-specific weights.
    Useful for validating configuration before running scoring.
    """
    print("\n" + "=" * 70)
    print("SECTOR-SPECIFIC WEIGHT CONFIGURATION")
    print("=" * 70)
    print(f"{'Sector':<16} {'ROE/PB':>8} {'ROE/PE':>8} {'Simple':>8} "
          f"{'Dividend':>10} {'Max Score':>10}")
    print("-" * 70)

    for sector in SECTOR_CRITERION_WEIGHTS:
        cw  = get_sector_weights(sector)
        dw  = DIVIDEND_WEIGHTS.get(sector, 0.0)
        mx  = compute_sector_max(sector)
        print(
            f"{sector:<16} {cw['roe_pb']:>8.1f} {cw['roe_pe']:>8.1f} "
            f"{cw['simple']:>8.1f} {dw:>10.1f} {mx:>10.1f}"
        )

    print("-" * 70)
    print(f"{'(fallback)':<16} "
          f"{DEFAULT_CRITERION_WEIGHTS['roe_pb']:>8.1f} "
          f"{DEFAULT_CRITERION_WEIGHTS['roe_pe']:>8.1f} "
          f"{DEFAULT_CRITERION_WEIGHTS['simple']:>8.1f} "
          f"{'0.0':>10} {'7.5':>10}")
    print("=" * 70)
    print(f"\nPayout ratio sustainability ceiling: {MAX_PAYOUT_RATIO:.0%}")
    print(f"Flag thresholds (normalised score):")
    for flag, threshold in FLAG_THRESHOLDS.items():
        print(f"  >= {threshold:.2f} → {flag}")
    print(f"  <  {min(FLAG_THRESHOLDS.values()):.2f} → Overvalued")
    print()


def summarise_scores(companies: pd.DataFrame) -> pd.DataFrame:
    """
    Print and return a summary of valuation flag distribution
    and average normalised scores by sector and flag.

    Parameters
    ----------
    companies : DataFrame after apply_valuation_flags() has been run

    Returns
    -------
    pd.DataFrame — grouped summary table
    """
    required = {"Sector", "Valuation Flag Comprehensive", "Normalised Valuation Score"}
    if not required.issubset(companies.columns):
        raise ValueError("Run apply_valuation_flags() before calling summarise_scores().")

    summary = (
        companies.groupby(["Sector", "Valuation Flag Comprehensive"])
        .agg(
            Count                    = ("Normalised Valuation Score", "count"),
            Avg_Normalised_Score     = ("Normalised Valuation Score", "mean"),
            Min_Normalised_Score     = ("Normalised Valuation Score", "min"),
            Max_Normalised_Score     = ("Normalised Valuation Score", "max"),
        )
        .round(4)
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("VALUATION FLAG DISTRIBUTION BY SECTOR")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Print weight configuration before running
    print_weight_summary()
    
def simulate_price_histories(tickers: list, sectors: np.ndarray) -> pd.DataFrame:

    all_histories = []

    for ticker, sector in zip(tickers, sectors):
        S0    = np.random.uniform(50, 150)
        mu    = SECTOR_PARAMS[sector]["mu"]
        sigma = SECTOR_PARAMS[sector]["sigma"]

        # ── Prices — length = DAYS ────────────────────────────────
        prices = [S0]                        # 1 element
        for _ in range(DAYS - 1):            # DAYS-1 more
            prices.append(
                prices[-1] * np.exp(
                    (mu - 0.5 * sigma ** 2) + sigma * np.random.normal()
                )
            )
        # prices length = DAYS ✓

        # ── Volumes — length = DAYS ───────────────────────────────
        vp        = SECTOR_VOLUME_PARAMS[sector]
        zero_prob = vp["zero_prob"]
        volumes   = []                       # starts empty

        for _ in range(DAYS):               # exactly DAYS iterations
            if np.random.random() < zero_prob:
                volumes.append(0)
            else:
                volumes.append(
                    max(0, np.random.normal(vp["mean"], vp["mean"] * 0.4))
                )
        # volumes length = DAYS ✓

        # ── DataFrame — all arrays must be DAYS length ────────────
        all_histories.append(pd.DataFrame({
            "Ticker": ticker,               # scalar — fine
            "Sector": sector,               # scalar — fine
            "Day":    range(1, DAYS + 1),   # length = DAYS ✓
            "Price":  prices,               # length = DAYS ✓
            "Volume": volumes               # length = DAYS ✓
        }))

    return pd.concat(all_histories, ignore_index=True)


def compute_returns(
    companies:    pd.DataFrame,
    expanded_df:  pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute daily returns, cumulative returns, and sector-normalised returns.

    Steps:
    ──────
    1. Daily % change per ticker
    2. Final compounded cumulative return per company (single value)
    3. Daily cumulative return trajectory per ticker (for charting)
    4. Merge final cumulative returns into fundamentals table
    5. Z-score normalise cumulative returns within each sector
       → Removes sector-level return drift so statistical tests compare
         valuation flags fairly across sectors with different mu values

    Parameters
    ----------
    companies   : fundamentals DataFrame
    expanded_df : price history DataFrame

    Returns
    -------
    tuple of (companies_with_returns, expanded_df_with_returns)
    """
    expanded_df = expanded_df.copy()

    # Step 1: Daily returns
    expanded_df["Return"] = expanded_df.groupby("Ticker")["Price"].pct_change()

    # Step 2: Final compounded cumulative return per ticker
    final_cumulative = (
        expanded_df.groupby("Ticker")["Return"]
        .apply(lambda r: (1 + r.dropna()).prod() - 1)
        .reset_index()
    )
    final_cumulative.columns = ["Ticker", "Cumulative Return"]

    # Step 3: Daily cumulative return trajectory (aligned via transform)
    expanded_df["Cumulative Return"] = (
        expanded_df.groupby("Ticker")["Return"]
        .transform(lambda r: (1 + r.fillna(0)).cumprod() - 1)
    )
# ── ADD THIS LINE — prevents column collision on merge ────────────────────
    if "Cumulative Return" in companies.columns:
        companies = companies.drop(columns=["Cumulative Return"])
# ─────────────────────────────────────────────────────────────────────────
    # Step 4: Merge into fundamentals
    companies = companies.merge(final_cumulative, on="Ticker", how="left")

    # Step 5: Z-score normalise within each sector
    companies["Normalized Return"] = companies.groupby("Sector")["Cumulative Return"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else np.nan
    )

    return companies, expanded_df


def save_outputs(
    companies:                 pd.DataFrame,
    expanded_df:               pd.DataFrame,
    avg_returns_by_sector_flag: pd.Series
) -> None:
    """Save all datasets to CSV for downstream analysis."""
    companies.to_csv("nse_value_dataset.csv", index=False)
    expanded_df.to_csv("all_price_histories.csv", index=False)
    avg_returns_by_sector_flag.reset_index().to_csv(
        "avg_returns_by_sector_flag.csv", index=False
    )
    print("Files saved: nse_value_dataset.csv, all_price_histories.csv, avg_returns_by_sector_flag.csv")


def plot_normalized_returns(companies: pd.DataFrame) -> None:
    """Boxplot of normalized returns by valuation flag."""
    order = ["Strongly Undervalued", "Moderately Undervalued", "Weakly Undervalued", "Overvalued"]
    existing_order = [f for f in order if f in companies["Valuation Flag Comprehensive"].unique()]

    plt.figure(figsize=(10, 6))
    sns.boxplot(
    x="Valuation Flag Comprehensive",
    y="Normalized Return",
    hue="Valuation Flag Comprehensive",
    data=companies,
    order=existing_order,
    palette=["#2ecc71", "#27ae60", "#f39c12", "#e74c3c"],
    legend=False
)
    plt.title("Normalized Returns by Valuation Flag (Sector-Adjusted)", fontsize=14)
    plt.xlabel("Valuation Flag")
    plt.ylabel("Normalized Return (Z-Score)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("normalized_returns_boxplot.png", dpi=150)
    plt.show()
    print("Plot saved: normalized_returns_boxplot.png")


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — STATISTICAL VALIDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_FREE_RATE   = 0.077 # CBK T-bill rate (annualized)
MIN_SAMPLE       = 3     # minimum observations per group for statistical tests
ALPHA            = 0.05  # significance threshold

# All flags representing undervalued stocks (scores 1, 2, 3)
# Used to correctly filter the dataset — "Undervalued" alone doesn't exist
UNDERVALUED_FLAGS = {"Strongly Undervalued", "Moderately Undervalued", "Weakly Undervalued"}


def validate_inputs(companies: pd.DataFrame) -> None:
    """
    Validate that required columns exist, are non-empty, and that
    Normalized Return is not entirely NaN (caused by zero-std sectors).
    Warns about any sectors where all normalized returns are NaN.
    """
    required_cols = {"Sector", "Valuation Flag Comprehensive", "Normalized Return"}
    missing = required_cols - set(companies.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if companies.empty:
        raise ValueError("Input DataFrame is empty.")
    if companies["Normalized Return"].isna().all():
        raise ValueError(
            "Normalized Return column is entirely NaN. "
            "Check that sector-level normalisation did not produce all-identical returns."
        )

    # Per-sector NaN warning
    bad_sectors = (
        companies.groupby("Sector")["Normalized Return"]
        .apply(lambda x: x.isna().all())
    )
    bad_sectors = bad_sectors[bad_sectors].index.tolist()
    if bad_sectors:
        print(f"[WARNING] Sectors with all-NaN Normalized Return (will be skipped): {bad_sectors}")


def compute_group_metrics(
    returns:       pd.Series,
    risk_free_rate: float = RISK_FREE_RATE
) -> dict:
    """
    Compute descriptive and risk-adjusted metrics for a return series.

    Metrics:
    ────────
    mean      : average normalized return
    std       : total volatility (standard deviation)
    n         : sample size
    sharpe    : (mean - risk_free_rate) / std
                → reward per unit of total risk
    sortino   : (mean - risk_free_rate) / downside_std
                → reward per unit of harmful downside risk only
                → preferred metric for NSE's thin, volatile market
    win_rate  : proportion of stocks beating the risk-free hurdle

    Parameters
    ----------
    returns        : pd.Series of normalized returns
    risk_free_rate : CBK T-bill rate (default 12%)
    """
    n = len(returns)
    if n == 0:
        return {
            "mean": None, "std": None, "n": 0,
            "sharpe": None, "sortino": None, "win_rate": None
        }

    mean   = returns.mean()
    std    = returns.std()
    excess = mean - risk_free_rate

    # Sharpe: penalises all volatility (upside and downside)
    sharpe = excess / std if std and std != 0 else None

    # Sortino: penalises only returns below the risk-free hurdle
    downside = returns[returns < risk_free_rate].std()
    sortino  = excess / downside if downside and downside != 0 else None

    # Win rate: proportion beating the CBK T-bill rate
    win_rate = float((returns > risk_free_rate).mean())

    return {
        "mean": mean, "std": std, "n": n,
        "sharpe": sharpe, "sortino": sortino, "win_rate": win_rate
    }


def cohens_d(g1: pd.Series, g2: pd.Series) -> float | None:
    """
    Compute Cohen's d effect size between two groups using pooled std.

    Answers: Is the separation between undervalued and overvalued returns
    large enough to be practically meaningful — not just statistically significant?

    With ~10 stocks per NSE sector, p-values alone are unreliable.
    Cohen's d provides the practical magnitude check.
    """
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return None
    pooled_var = ((n1 - 1) * g1.std() ** 2 + (n2 - 1) * g2.std() ** 2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    return (g1.mean() - g2.mean()) / pooled_std if pooled_std != 0 else None


def interpret_cohens_d(d: float | None) -> str:
    """
    Return a human-readable label for Cohen's d magnitude.

    Thresholds: <0.2 Negligible | 0.2–0.5 Small | 0.5–0.8 Medium | >0.8 Large
    """
    if d is None:
        return "N/A"
    d = abs(d)
    if d < 0.2: return "Negligible"
    if d < 0.5: return "Small"
    if d < 0.8: return "Medium"
    return "Large"

def check_normality(series: pd.Series) -> bool:
    """
    Run D'Agostino-Pearson normality test on a return series.
    Returns True if normality cannot be rejected (p >= 0.05).
    Requires n >= 8 for reliable results — returns False otherwise.

    NSE stock returns often violate normality due to fat tails and skew
    from thin liquidity, making this test critical for test selection.
    """
    if len(series) < 8:
        return False
    _, p = normaltest(series)
    return p >= 0.05

def run_statistical_tests(
    undervalued: pd.Series,
    overvalued:  pd.Series,
    min_sample:  int   = MIN_SAMPLE,
    alpha:       float = ALPHA
) -> dict:
    """
    Run both Welch's t-test and Mann-Whitney U test.
    Select the preferred test based on normality of both groups.

    Test Selection Logic:
    ─────────────────────
    Both groups normal  → Welch's t-test (parametric, more powerful)
    Either non-normal   → Mann-Whitney U (non-parametric, distribution-free)

    Both results are always recorded regardless of which is preferred,
    giving a complete picture for reporting and audit purposes.

    Parameters
    ----------
    undervalued : normalized returns of undervalued stocks
    overvalued  : normalized returns of overvalued stocks
    min_sample  : minimum n per group to run tests (default 5)
    alpha       : significance threshold (default 0.05)
    """
    uv_n, ov_n = len(undervalued), len(overvalued)

    if uv_n < min_sample or ov_n < min_sample:
        return {
            "t_stat": None, "p_value_t": None,
            "u_stat": None, "p_value_u": None,
            "preferred_test": "None",
            "significance":   "Insufficient Data"
        }

    # Welch's t-test (does not assume equal variances)
    t_stat, p_val_t = ttest_ind(undervalued, overvalued, equal_var=False)

    # Mann-Whitney U (non-parametric, rank-based)
    u_stat, p_val_u = mannwhitneyu(undervalued, overvalued, alternative="two-sided")

    # Select preferred test based on normality
    uv_normal      = check_normality(undervalued)
    ov_normal      = check_normality(overvalued)
    both_normal    = uv_normal and ov_normal
    preferred_test = "Welch's t-test" if both_normal else "Mann-Whitney U"
    preferred_p    = p_val_t if both_normal else p_val_u

    significance = "Significant" if preferred_p < alpha else "Not Significant"

    return {
        "t_stat":         round(t_stat,  4),
        "p_value_t":      round(p_val_t, 4),
        "u_stat":         round(u_stat,  4),
        "p_value_u":      round(p_val_u, 4),
        "preferred_test": preferred_test,
        "significance":   significance
    }

def run_kruskal_wallis(companies: pd.DataFrame) -> dict:
    groups = []
    labels = [
        "Strongly Undervalued",
        "Moderately Undervalued",
        "Weakly Undervalued",
        "Overvalued"
    ]

    for label in labels:
        group = companies[
            companies["Valuation Flag Comprehensive"] == label
        ]["Normalized Return"].dropna()
        if len(group) >= 2:
            groups.append(group)

    if len(groups) < 2:
        return {
            "h_statistic":  None,
            "p_value":      None,
            "significance": "Insufficient Groups",
            "groups_tested": len(groups)
        }

    h_stat, p_val = kruskal(*groups)

    return {
        "h_statistic":   round(h_stat, 4),
        "p_value":       round(p_val,  4),
        "significance":  "Significant" if p_val < ALPHA else "Not Significant",
        "groups_tested": len(groups)
    }

def runs_test(returns: pd.Series) -> dict:
    if len(returns) < 8:
        return {"pattern": "Insufficient Data"}

    median = returns.median()
    signs  = (returns > median).astype(int)

    # Count runs — a run is a sequence of consecutive same-sign values
    runs = 1 + (signs.diff().abs() > 0).sum()
    n1   = signs.sum()
    n2   = len(signs) - n1
    n    = len(signs)

    expected_runs = (2 * n1 * n2 / n) + 1
    variance_runs = (
        (2 * n1 * n2 * (2 * n1 * n2 - n)) /
        (n ** 2 * (n - 1))
    )

    z_score = (runs - expected_runs) / np.sqrt(variance_runs)

    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    pattern = (
        "Trending — undervaluation persists"
        if p_value < 0.05 and runs < expected_runs
        else "Mean-reverting — quick correction"
        if p_value < 0.05 and runs > expected_runs
        else "Random — no clear pattern"
    )

    return {
        "runs":          int(runs),
        "expected_runs": round(expected_runs, 2),
        "z_score":       round(z_score, 4),
        "p_value":       round(p_value, 4),
        "pattern":       pattern
    }

def compute_amihud_ratio(price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Amihud Ratio = mean(|Return| / Volume_KSh)

    Interpretation:
    Low  ratio → liquid  → 1% return needs large volume → price is real
    High ratio → illiquid → 1% return needs tiny volume → price is paper
    """
    def amihud_per_ticker(group):
        valid = group.dropna(subset=["Return", "Volume"])
        valid = valid[valid["Volume"] > 0]
        if len(valid) == 0:
            return np.nan
        return (valid["Return"].abs() / valid["Volume"]).mean()

    amihud = (
        price_history.groupby("Ticker")
        .apply(amihud_per_ticker)
        .reset_index()
    )
    amihud.columns = ["Ticker", "Amihud_Ratio"]
    return amihud

def apply_liquidity_veto(companies: pd.DataFrame) -> pd.DataFrame:
    """
    Downgrade any Strongly or Moderately Undervalued stock
    whose Amihud ratio exceeds the liquidity veto threshold.
    ...
    """
    if "Amihud_Ratio" not in companies.columns:
        print("[WARNING] Amihud_Ratio column not found — veto skipped")
        return companies

    companies = companies.copy()

    veto_mask = (
        (companies["Amihud_Ratio"] > AMIHUD_VETO_THRESHOLD) &
        (companies["Valuation Flag Comprehensive"].isin([
            "Strongly Undervalued",
            "Moderately Undervalued"
        ]))
    )

    veto_count = veto_mask.sum()

    if veto_count > 0:
        print(f"\n[LIQUIDITY VETO] {veto_count} stocks downgraded "
              f"to {VETO_DOWNGRADE_FLAG}:")
        vetoed = companies[veto_mask][
            ["Ticker", "Sector", "Valuation Flag Comprehensive",
             "Amihud_Ratio", "Normalised Valuation Score"]
        ]
        print(vetoed.to_string(index=False))

        companies.loc[veto_mask, "Valuation Flag Comprehensive"] = (
            VETO_DOWNGRADE_FLAG
        )

        companies.loc[veto_mask, "Score Breakdown"] = (
            companies.loc[veto_mask, "Score Breakdown"]
            + f" | VETO=Liquidity(Amihud>{AMIHUD_VETO_THRESHOLD:.0e})"
        )
    else:
        print("\n[LIQUIDITY VETO] No stocks vetoed — all pass liquidity check")

    return companies

def compute_liquidity_adjusted_sharpe(
    returns:          pd.Series,
    zero_volume_days: int,
    total_days:       int,
    risk_free_rate:   float = RISK_FREE_RATE
) -> float | None:
    std = returns.std()
    if std is None or std == 0:
        return None
    raw_sharpe        = (returns.mean() - risk_free_rate) / std
    liquidity_penalty = 1 - (zero_volume_days / total_days)
    return raw_sharpe * liquidity_penalty

def compute_jensens_alpha(
    portfolio_returns: pd.Series,
    market_returns:    pd.Series,
    risk_free_rate:    float = RISK_FREE_RATE,
    market_caps:       pd.Series = None
) -> dict:
    """
    Compute Jensen's Alpha using CAPM framework.
    Alpha = Rp - [Rf + Beta * (Rm - Rf)]
    Beta fixed at 1.0 for equal-weighted universe proxy.

    If market_caps provided uses market-cap weighted market
    return to better approximate NSE Safaricom-dominated
    structure. Otherwise falls back to equal-weighted mean.
    """
    if len(portfolio_returns) < 3:
        return {"alpha": None, "beta": None,
                "interpretation": "Insufficient Data"}

    # ── Market return ─────────────────────────────────────────────
    if market_caps is not None and len(market_caps) == len(market_returns):
        weights      = market_caps / market_caps.sum()
        market_mean  = (market_returns * weights).sum()
        proxy_method = "Market-Cap Weighted"
    else:
        market_mean  = market_returns.mean()
        proxy_method = "Equal-Weighted"

    portfolio_mean  = portfolio_returns.mean()
    beta            = 1.0
    expected_return = risk_free_rate + beta * (market_mean - risk_free_rate)
    alpha           = portfolio_mean - expected_return

    interpretation = (
        "Outperforms market risk-adjusted"
        if alpha > 0
        else "Underperforms — buy NASI index instead"
    )

    return {
        "alpha":          round(alpha, 4),
        "beta":           round(beta,  4),
        "proxy_method":   proxy_method,
        "interpretation": interpretation
    }
 
def compute_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float | None:
    """
    IR = Active Return / Tracking Error

    Active Return  = mean(portfolio) - mean(benchmark)
    Tracking Error = std(portfolio returns - benchmark returns)

    IR > 0.5 = consistently beating the index
    IR > 1.0 = excellent — rare in any market
    """
    active_returns = portfolio_returns.values - benchmark_returns.mean()
    tracking_error = active_returns.std()

    if tracking_error == 0:
        return None

    return round(active_returns.mean() / tracking_error, 4)

def assign_risk_flag(row: pd.Series) -> str:
    """
    Standalone risk quality flag surfaced as its own output column.

    Uses Sortino difference as the primary signal (preferred because it
    only penalises harmful downside volatility, not upside variance).
    Falls back to Sharpe difference if Sortino is unavailable.

    Labels:
    ────────────────────────────────────────────────────────────────────
    Strong Downside Protection : Sortino diff > +0.5
    Mild Downside Protection   : Sortino diff > 0
    Mild Downside Risk         : Sortino diff > -0.5
    High Downside Risk         : Sortino diff ≤ -0.5
    Risk-Adjusted Positive     : Sharpe diff > 0  (Sortino unavailable)
    Risk-Adjusted Negative     : Sharpe diff ≤ 0  (Sortino unavailable)
    No Risk Data               : both diffs are None
    """
    sd = row.get("Sharpe Difference")
    so = row.get("Sortino Difference")

    if sd is None and so is None:
        return "No Risk Data"

    if so is not None:
        if so > 0.5:  return "Strong Downside Protection"
        if so > 0:    return "Mild Downside Protection"
        if so > -0.5: return "Mild Downside Risk"
        return "High Downside Risk"

    if sd is not None:
        if sd > 0: return "Risk-Adjusted Positive"
        return "Risk-Adjusted Negative"

    return "No Risk Data"

def assign_verdict(row: pd.Series) -> str:
    """
    Assign a plain-language verdict combining statistical significance,
    return direction, and risk-adjusted qualification.

    A full 'Algorithm Validated' verdict requires BOTH:
      1. Statistically significant result (preferred test p < alpha)
      2. Undervalued mean > Overvalued mean (raw outperformance)
      3. At least one of: positive Sortino diff OR positive Sharpe diff
         (risk-adjusted outperformance — not just raw return)

    This prevents a misleading 'Validated' label when undervalued stocks
    earn more but take on disproportionately higher risk to do so —
    a critical distinction for NSE investors operating in a thin market.

    Four Outcome Quadrants:
    ────────────────────────────────────────────────────────────────────────────
    Outperforms + risk-adjusted   → Algorithm Validated
    Outperforms + NOT risk-adj    → Caution (raw gain, hidden risk)
    Underperforms + risk-adjusted → Mixed (safer but lower raw return)
    Underperforms + NOT risk-adj  → Algorithm Contradicted
    """
    sig          = row["Significance"]
    uv_m         = row["Undervalued Mean"]
    ov_m         = row["Overvalued Mean"]
    d            = row["Cohen's D Magnitude"]
    sharpe_diff  = row.get("Sharpe Difference")
    sortino_diff = row.get("Sortino Difference")

    # Inconclusive cases — exit early
    if sig == "Insufficient Data":
        return "Inconclusive — Insufficient Data"
    if sig == "Not Significant":
        return "Inconclusive — No Significant Difference"
    if uv_m is None or ov_m is None:
        return "Inconclusive — Missing Data"

    outperforms = uv_m > ov_m

    # Risk-adjusted qualification: Sortino primary, Sharpe secondary
    sortino_positive = sortino_diff is not None and sortino_diff > 0
    sharpe_positive  = sharpe_diff  is not None and sharpe_diff  > 0
    risk_adjusted    = sortino_positive or sharpe_positive

    if outperforms and risk_adjusted:
        return f"Algorithm Validated — Risk-Adjusted Outperformance ({d} Effect)"
    elif outperforms and not risk_adjusted:
        return f"Caution — Raw Outperformance Only, Risk-Adjusted Return Inferior ({d} Effect)"
    elif not outperforms and risk_adjusted:
        return f"Mixed — Undervalued Stocks Safer But Lower Raw Return ({d} Effect)"
    else:
        return f"Algorithm Contradicted — Underperforms on Return and Risk ({d} Effect)"

def compute_summary_row(companies: pd.DataFrame, risk_free_rate: float) -> dict:
    """
    Compute aggregate metrics pooling all sectors together.

    Answers the top-level question: does the algorithm work across the
    NSE as a whole, independent of sector-level variation?

    FIX APPLIED: Uses != 'Overvalued' to correctly capture all three
    undervalued sub-flags. Filtering for the literal string 'Undervalued'
    returns zero rows since the actual flags are 'Strongly Undervalued',
    'Moderately Undervalued', and 'Weakly Undervalued'.
    """
    # Correctly capture all undervalued sub-flags
    all_uv = companies[
        companies["Valuation Flag Comprehensive"] != "Overvalued"
    ]["Normalized Return"].dropna()

    all_ov = companies[
        companies["Valuation Flag Comprehensive"] == "Overvalued"
    ]["Normalized Return"].dropna()

    uv_metrics = compute_group_metrics(all_uv, risk_free_rate)
    ov_metrics = compute_group_metrics(all_ov, risk_free_rate)
    tests      = run_statistical_tests(all_uv, all_ov)
    d          = cohens_d(all_uv, all_ov)

    sharpe_diff = (
        uv_metrics["sharpe"]  - ov_metrics["sharpe"]
        if uv_metrics["sharpe"]  is not None and ov_metrics["sharpe"]  is not None else None
    )
    sortino_diff = (
        uv_metrics["sortino"] - ov_metrics["sortino"]
        if uv_metrics["sortino"] is not None and ov_metrics["sortino"] is not None else None
    )

    summary = {
        "Sector":               "── ALL SECTORS ──",
        "Undervalued Mean":     uv_metrics["mean"],
        "Undervalued Std":      uv_metrics["std"],
        "Undervalued N":        uv_metrics["n"],
        "Undervalued Sharpe":   uv_metrics["sharpe"],
        "Undervalued Sortino":  uv_metrics["sortino"],
        "Undervalued Win Rate": uv_metrics["win_rate"],
        "Overvalued Mean":      ov_metrics["mean"],
        "Overvalued Std":       ov_metrics["std"],
        "Overvalued N":         ov_metrics["n"],
        "Overvalued Sharpe":    ov_metrics["sharpe"],
        "Overvalued Sortino":   ov_metrics["sortino"],
        "Overvalued Win Rate":  ov_metrics["win_rate"],
        "Sharpe Difference":    sharpe_diff,
        "Sortino Difference":   sortino_diff,
        "T-Statistic":          tests["t_stat"],
        "P-Value (t-test)":     tests["p_value_t"],
        "U-Statistic":          tests["u_stat"],
        "P-Value (MWU)":        tests["p_value_u"],
        "Preferred Test":       tests["preferred_test"],
        "Significance":         tests["significance"],
        "Cohen's D":            round(d, 4) if d is not None else None,
        "Cohen's D Magnitude":  interpret_cohens_d(d),
        "Risk Flag":            "── See Per-Sector Results ──",
        "Verdict":              "── See Per-Sector Results ──"
    }

    return summary

def validate_sectors(
    companies:      pd.DataFrame,
    min_sample:     int   = MIN_SAMPLE,
    risk_free_rate: float = RISK_FREE_RATE,
    alpha:          float = ALPHA
) -> pd.DataFrame:
    """
    Validate the NSE stock valuation algorithm sector-by-sector.

    For each sector, splits stocks into undervalued (score 1–3) vs
    overvalued (score 0) and runs a full statistical comparison:

    Metrics per group:
      - Mean, Std, N
      - Sharpe ratio  (CBK risk-free rate adjusted)
      - Sortino ratio (downside volatility only)
      - Win rate vs risk-free hurdle

    Differential metrics:
      - Sharpe Difference
      - Sortino Difference

    Statistical tests (both always run):
      - Welch's t-test  (parametric)
      - Mann-Whitney U  (non-parametric)
      - Preferred test selected by D'Agostino-Pearson normality check

    Effect size:
      - Cohen's d + magnitude label

    Output columns:
      - Risk Flag  : Sortino-primary risk quality label
      - Verdict    : risk-adjusted plain-language conclusion

    FIX APPLIED: undervalued group uses != 'Overvalued' to correctly
    capture Strongly / Moderately / Weakly Undervalued stocks.

    Parameters
    ----------
    companies      : DataFrame with NSE fundamentals and Normalized Return
    min_sample     : minimum group size for statistical tests (default 5)
    risk_free_rate : CBK T-bill rate (default 0.12)
    alpha          : significance level (default 0.05)

    Returns
    -------
    pd.DataFrame — one row per sector + aggregate summary row
    """
    validate_inputs(companies)

    results = []

    for sector, group in companies.groupby("Sector"):
        group = group.copy()

        # ── FIXED: capture all undervalued sub-flags via != 'Overvalued' ──────
        undervalued = group[
            group["Valuation Flag Comprehensive"] != "Overvalued"
        ]["Normalized Return"].dropna()

        overvalued = group[
            group["Valuation Flag Comprehensive"] == "Overvalued"
        ]["Normalized Return"].dropna()

        # ── Metrics ───────────────────────────────────────────────────────────
        uv = compute_group_metrics(undervalued, risk_free_rate)
        ov = compute_group_metrics(overvalued,  risk_free_rate)

        sharpe_diff = (
            uv["sharpe"]  - ov["sharpe"]
            if uv["sharpe"]  is not None and ov["sharpe"]  is not None else None
        )
        sortino_diff = (
            uv["sortino"] - ov["sortino"]
            if uv["sortino"] is not None and ov["sortino"] is not None else None
        )

        # ── Statistical Tests ─────────────────────────────────────────────────
        tests = run_statistical_tests(undervalued, overvalued, min_sample, alpha)

        # ── Effect Size ───────────────────────────────────────────────────────
        d = cohens_d(undervalued, overvalued)

        row = {
            "Sector":               sector,
            # Undervalued group
            "Undervalued Mean":     round(uv["mean"],     4) if uv["mean"]     is not None else None,
            "Undervalued Std":      round(uv["std"],      4) if uv["std"]      is not None else None,
            "Undervalued N":        uv["n"],
            "Undervalued Sharpe":   round(uv["sharpe"],   4) if uv["sharpe"]   is not None else None,
            "Undervalued Sortino":  round(uv["sortino"],  4) if uv["sortino"]  is not None else None,
            "Undervalued Win Rate": round(uv["win_rate"], 4) if uv["win_rate"] is not None else None,
            # Overvalued group
            "Overvalued Mean":      round(ov["mean"],     4) if ov["mean"]     is not None else None,
            "Overvalued Std":       round(ov["std"],      4) if ov["std"]      is not None else None,
            "Overvalued N":         ov["n"],
            "Overvalued Sharpe":    round(ov["sharpe"],   4) if ov["sharpe"]   is not None else None,
            "Overvalued Sortino":   round(ov["sortino"],  4) if ov["sortino"]  is not None else None,
            "Overvalued Win Rate":  round(ov["win_rate"], 4) if ov["win_rate"] is not None else None,
            # Differential metrics
            "Sharpe Difference":    round(sharpe_diff,    4) if sharpe_diff    is not None else None,
            "Sortino Difference":   round(sortino_diff,   4) if sortino_diff   is not None else None,
            # Statistical tests
            "T-Statistic":          tests["t_stat"],
            "P-Value (t-test)":     tests["p_value_t"],
            "U-Statistic":          tests["u_stat"],
            "P-Value (MWU)":        tests["p_value_u"],
            "Preferred Test":       tests["preferred_test"],
            "Significance":         tests["significance"],
            # Effect size
            "Cohen's D":            round(d, 4) if d is not None else None,
            "Cohen's D Magnitude":  interpret_cohens_d(d),
        }

        # Risk Flag and Verdict added after row is fully assembled
        row["Risk Flag"] = assign_risk_flag(pd.Series(row))
        row["Verdict"]   = assign_verdict(pd.Series(row))

        results.append(row)

   # ── Build Output DataFrame ────────────────────────────────────────────────
    sector_results = pd.DataFrame(results)

    # ── NEW: Kruskal-Wallis across all four tiers ─────────────────────────────
    # Runs once on the full dataset — not per sector
    kw_result = run_kruskal_wallis(companies)

    # ── Append aggregate summary row ─────────────────────────────────────────
    summary = compute_summary_row(companies, risk_free_rate)

    # Add KW results to the summary row
    summary["KW H-Statistic"] = kw_result["h_statistic"]
    summary["KW P-Value"]     = kw_result["p_value"]
    summary["KW Significance"]= kw_result["significance"]
    summary["KW Groups Tested"]= kw_result["groups_tested"]

    sector_results = pd.concat(
        [sector_results, pd.DataFrame([summary])],
        ignore_index=True
    )

    return sector_results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Orchestrate the full pipeline
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 60)
    print("PHASE 1: Data Generation & Valuation Flagging")
    print("=" * 60)

    # ── Step 1: Generate companies — fundamentals only ────────────────────────
    companies = generate_companies()

    # ── Step 2: Simulate prices ───────────────────────────────────────────────
    expanded_df = simulate_price_histories(
        companies["Ticker"].tolist(),
        companies["Sector"].values
    )

    # ── Step 3: Compute returns — adds Cumulative Return to companies ─────────
    companies, expanded_df = compute_returns(companies, expanded_df)
    # companies now has Cumulative Return and Normalized Return columns

    # ── Step 4: Compute Amihud — needs Volume column in expanded_df ───────────
    if "Volume" in expanded_df.columns:
        amihud_results = compute_amihud_ratio(expanded_df)
        companies = companies.merge(amihud_results, on="Ticker", how="left")
        print("\nAmihud ratios computed and merged into company data")

    # ── Step 5: Score companies — first pass with default thresholds ──────────
    companies = apply_valuation_flags(companies)

    print(f"\nCompanies generated: {len(companies)}")
    print("\nPre-calibration flag distribution:")
    print(companies["Valuation Flag Comprehensive"].value_counts())

    print("\nPre-calibration score distribution:")
    print(companies["Normalised Valuation Score"].describe().round(4))

    # ── Step 6: Calibrate thresholds ─────────────────────────────────────────
    suggested_thresholds = calibrate_thresholds(companies)
    print("\nSuggested thresholds:", suggested_thresholds)

    companies = apply_valuation_flags(
        companies, thresholds=suggested_thresholds
    )

    print("\nPost-calibration flag distribution (before veto):")
    print(companies["Valuation Flag Comprehensive"].value_counts())

    # ── Step 7: Apply Liquidity Veto ──────────────────────────────────────────
    if "Amihud_Ratio" in companies.columns:
        companies = apply_liquidity_veto(companies)
        print("\nPost-veto flag distribution:")
        print(companies["Valuation Flag Comprehensive"].value_counts())

    # ── Step 8: Compute average returns ───────────────────────────────────────
    avg_returns_by_sector_flag = companies.groupby(
        ["Sector", "Valuation Flag Comprehensive"]
    )["Cumulative Return"].mean()

    print("\nAverage Cumulative Return by Sector and Valuation Flag:")
    print(avg_returns_by_sector_flag.round(4))

    save_outputs(companies, expanded_df, avg_returns_by_sector_flag)
  
   
    # ── Phase 2: Statistical Validation ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 2: Statistical Validation")
    print("=" * 60)

    sector_results = validate_sectors(
        companies,
        min_sample     = MIN_SAMPLE,
        risk_free_rate = RISK_FREE_RATE,
        alpha          = ALPHA
    )

    # Display full results
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    
    print("\nSector Validation Results:")
    print(sector_results.to_string(index=False))

    # Save validation results
    sector_results.to_csv("sector_validation_results.csv", index=False)
    print("\nValidation results saved: sector_validation_results.csv")

# ══ ADD ALL ENHANCEMENTS HERE ════════════════════════════════
    print("\n" + "─" * 60)
    print("KRUSKAL-WALLIS H-TEST (all four valuation tiers)")
    print("─" * 60)
    kw_result = run_kruskal_wallis(companies)
    print(f"  H-statistic  : {kw_result['h_statistic']}")
    print(f"  p-value      : {kw_result['p_value']}")
    print(f"  Groups tested: {kw_result['groups_tested']}")
    print(f"  Result       : {kw_result['significance']}")

    print("\n" + "─" * 60)
    print("RUNS TEST (undervalued group return pattern)")
    print("─" * 60)
    all_uv_returns = companies[
        companies["Valuation Flag Comprehensive"] != "Overvalued"
        ]["Normalized Return"].dropna()
    runs_result = runs_test(all_uv_returns)
    print(f"  Runs observed : {runs_result.get('runs', 'N/A')}")
    print(f"  Runs expected : {runs_result.get('expected_runs', 'N/A')}")
    print(f"  Z-score       : {runs_result.get('z_score', 'N/A')}")
    print(f"  p-value       : {runs_result.get('p_value', 'N/A')}")
    print(f"  Pattern       : {runs_result.get('pattern', 'N/A')}")

    print("\n" + "─" * 60)
    print("JENSEN'S ALPHA (undervalued vs market proxy)")
    print("─" * 60)
    market_proxy = companies["Normalized Return"].dropna()
    alpha_result = compute_jensens_alpha(
        all_uv_returns,
        market_proxy,
        RISK_FREE_RATE,
        market_caps=companies["Market Cap (KES B)"]    
    )
    print(f"  Alpha         : {alpha_result['alpha']}")
    print(f"  Beta          : {alpha_result['beta']}")
    print(f"  Proxy method  : {alpha_result['proxy_method']}")    
    print(f"  Interpretation: {alpha_result['interpretation']}")

    print("\n" + "─" * 60)
    print("INFORMATION RATIO (consistency of outperformance)")
    print("─" * 60)
    ir = compute_information_ratio(all_uv_returns, market_proxy)
    print(f"  IR : {ir}")
    if ir is not None:
            rating = (
                "Excellent" if ir > 1.0
                else "Good"     if ir > 0.5
                else "Positive" if ir > 0.0
                else "Negative — consider passive NASI index instead"
            )
            print(f"  Rating : {rating}")

    if "Amihud_Ratio" in companies.columns:
        print("\n" + "─" * 60)
        print("AMIHUD ILLIQUIDITY RATIO (per valuation flag)")
        print("─" * 60)

        nan_count = companies["Amihud_Ratio"].isna().sum()
        if nan_count > 0:
            print(f"  [WARNING] {nan_count} companies missing Amihud data")

        amihud_by_flag = (
            companies
            .groupby("Valuation Flag Comprehensive")["Amihud_Ratio"]
            .mean()
        )

        LIQUID_THRESHOLD   = 1e-6
        MODERATE_THRESHOLD = 1e-5

        for flag, ratio in amihud_by_flag.items():
            if pd.isna(ratio):
                print(f"  {flag:<25} N/A — no liquidity data")
                continue
            liquidity_label = (
                "LIQUID"         if ratio < LIQUID_THRESHOLD
                else "MODERATE"  if ratio < MODERATE_THRESHOLD
                else "ILLIQUID"
            )
            print(f"  {flag:<25} {ratio:.2e}   {liquidity_label}")

        print(f"\n  Lower = more liquid. High Amihud = paper gains only.")
        print(f"  Thresholds: LIQUID < {LIQUID_THRESHOLD:.0e} | "
              f"MODERATE < {MODERATE_THRESHOLD:.0e} | "
              f"ILLIQUID >= {MODERATE_THRESHOLD:.0e}")
        print(f"  Note: Volume assumed in raw KES. "
              f"Adjust thresholds if units change at Milestone 4.")
    # ══ END OF ENHANCEMENTS ══════════════════════════════════════

    # Verdict summary 
    print("\n" + "=" * 60)
    print("VERDICT SUMMARY")
    print("=" * 60)
    verdict_cols = ["Sector", "Risk Flag", "Verdict"]
    print(sector_results[verdict_cols].to_string(index=False))

# ── Phase 3: Visualisation ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3: Visualisation")
    print("=" * 60)

    plot_normalized_returns(companies)


