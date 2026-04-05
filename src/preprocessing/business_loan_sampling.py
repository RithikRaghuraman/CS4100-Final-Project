"""
Script to sample business loan data from the 2010-2025 time period.
Samples as many defaulted loans as possible per fiscal year with the rest 
filled with fully paid loans. The resulting CSV is kept under 10MB for 
quicker development. 
"""

import pandas as pd
from pathlib import Path


TARGET_BYTES = 10 * 1024 * 1024
OUTPUT_NAME = "business_loans_2010_2025.csv"


def main():
    project_root = Path(__file__).parent.parent.parent
    input_path = project_root / "data" / "foia-504-fy2010-present-asof-251231.csv"
    output_path = project_root / "data" / OUTPUT_NAME

    loans = pd.read_csv(input_path, low_memory=False)

    file_size = input_path.stat().st_size
    bytes_per_row = file_size / len(loans)

    # Filter to only PIF and CHGOFF loans
    loans = loans[loans["loanstatus"].isin(["PIF", "CHGOFF"])]
    max_total_rows = int(TARGET_BYTES / bytes_per_row)

    chgoff = loans[loans["loanstatus"] == "CHGOFF"]
    pif = loans[loans["loanstatus"] == "PIF"]

    n_pif = max(0, max_total_rows - len(chgoff))
    n_pif = min(n_pif, len(pif))

    sampled = pd.concat([chgoff, pif.sample(n=n_pif, random_state=10)]).reset_index(drop=True)

    sampled.to_csv(output_path, index=False)

    actual_mb = output_path.stat().st_size / 1024 / 1024
    print(f"CHGOFF: {len(chgoff):,}  PIF sampled: {n_pif:,}  Total: {len(sampled):,}")
    print(f"Saved → {output_path.name}  ({actual_mb:.2f} MB)")


if __name__ == "__main__":
    main()
