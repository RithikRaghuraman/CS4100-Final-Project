"""
Script to sample business loan data from the 2010-2025 time period.
Samples an equal number of loans per fiscal year so the resulting CSV is under 10 MB.
"""

import pandas as pd
from pathlib import Path


TARGET_BYTES = 10 * 1024 * 1024  # 10 MB
OUTPUT_NAME = "business_loans_2010_2025.csv"


def main():
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "foia-504-fy2010-present-asof-251231.csv"
    output_path = project_root / "data" / OUTPUT_NAME

    loans = pd.read_csv(input_path, low_memory=False)

    total_rows = len(loans)
    file_size = input_path.stat().st_size
    bytes_per_row = file_size / total_rows
    num_years = loans["approvalfiscalyear"].nunique()

    # Max rows that fit in 10 MB, divided equally across years
    max_total_rows = int(TARGET_BYTES / bytes_per_row)
    n_per_year = max_total_rows // num_years

    groups = []
    for _, group in loans.groupby("approvalfiscalyear"):
        print(group["approvalfiscalyear"], len(group))
        n = min(n_per_year, len(group))
        groups.append(group.sample(n=n, random_state=10))

    sampled = pd.concat(groups).reset_index(drop=True)

    sampled.to_csv(output_path, index=False)

    actual_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Saved {len(sampled):,} rows → {output_path.name}  ({actual_mb:.2f} MB)")


if __name__ == "__main__":
    main()
