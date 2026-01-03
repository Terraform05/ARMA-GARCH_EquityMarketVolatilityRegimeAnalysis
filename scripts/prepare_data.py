from __future__ import annotations
from src.data_prep import prepare_aligned_dataset
from src.data_fetch import download_yahoo_data
from src.config import END_DATE, OUTPUT_CSV, SPX_TICKER, START_DATE, VIX_TICKER

from pathlib import Path
import sys

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_data_prep(
    start: str = START_DATE,
    end: str = END_DATE,
    output: str | Path = OUTPUT_CSV,
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(total=2, desc="Downloading Yahoo data") as progress:
        spx_df = download_yahoo_data(SPX_TICKER, start, end)
        progress.update(1)
        vix_df = download_yahoo_data(VIX_TICKER, start, end)
        progress.update(1)

    aligned = prepare_aligned_dataset(spx_df, vix_df)
    aligned.to_csv(output_path, index=False)

    print(f"Wrote {len(aligned):,} rows to {output_path}")
    return output_path


if __name__ == "__main__":
    run_data_prep()
