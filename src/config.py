from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

START_DATE = "2010-01-01"
# Default to the most recent calendar day; data fetch will align to last trading day.
END_DATE = date.today().isoformat()

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

OUTPUT_CSV = PROCESSED_DIR / "spx_vix_aligned.csv"

MODEL_VARIANTS_DIR = PROJECT_ROOT / "reports" / "modeling_variants"
BEST_VARIANT_FILE = MODEL_VARIANTS_DIR / "data" / "best_variant.txt"
VARIANT_METRICS_FILE = MODEL_VARIANTS_DIR / "data" / "variant_realized_metrics.csv"

# Variant selection mode: "bic" (default) or "tracking"
VARIANT_SELECTION = "tracking" #"bic"
