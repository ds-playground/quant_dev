# =============================================================================
#  config.py  —  Central configuration for the price-return analysis framework
# =============================================================================
import os

# ── Paths (relative to this file's location) ─────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEV_FILES    = os.path.join(PROJECT_ROOT, "dev_files")

# ── Ticker / data ─────────────────────────────────────────────────────────────
TICKER       = "AAPL"          # symbol shown in chart titles
N_DAYS       = 500             # number of simulated trading days
START_DATE   = "2022-01-03"    # simulation / real-data start

# ── Analysis parameters ───────────────────────────────────────────────────────
WIN_THRESHOLD  =  0.5               # min daily return (%) counted as a win
LOSS_THRESHOLD = -0.5               # max daily return (%) counted as a loss
WINDOWS        = [2, 3, 5]          # consecutive-day windows to scan
CUM_THRESHOLDS = [0.5, 1.0, 2.0]   # cumulative return thresholds (%)

# ── Chart appearance ──────────────────────────────────────────────────────────
COLOR_WIN    = "#1D9E75"
COLOR_LOSS   = "#D85A30"
COLOR_PRICE  = "#378ADD"
COLOR_CUM    = ["#378ADD", "#7F77DD", "#D4537E"]
CHART_BG     = "white"

# ── Output file names (saved inside DEV_FILES) ────────────────────────────────
FILE_STREAK_CSV    = "streak_summary.csv"
FILE_CUM_CSV       = "cumulative_summary.csv"
FILE_PRICE_HTML    = "chart_price_return.html"
FILE_DIST_HTML     = "chart_distribution.html"
FILE_STREAK_HTML   = "chart_streaks.html"
FILE_CUM_HTML      = "chart_cumulative.html"
FILE_TIMELINE_HTML = "chart_timeline.html"
