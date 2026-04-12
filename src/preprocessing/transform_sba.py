"""
ML transformation pipeline for cleaned SB data
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

VALID_STATUSES = ["PIF", "CHGOFF"]

_NUMERIC_COLS = ["grossapproval", "thirdpartydollars", "jobssupported", "leverageratio"]

_PASSTHROUGH_COLS = ["terminmonths", "approvalyear"]

_CAT_COLS = ["borrstate", "processingmethod", "businesstype", "naics_sector"]

_BINARY_COLS = ["collateralind"]


def stratified_split(df, train_frac = 0.70, val_frac = 0.15, seed = 42):
    """
    Label class aware split into train/val/test
    Returns tuple of (train, val, test) dfs
    """
    labeled = df[df["loanstatus"].isin(VALID_STATUSES)].copy()

    defaults = labeled[labeled["loanstatus"] == "CHGOFF"].sample(frac=1, random_state=seed)
    paid = labeled[labeled["loanstatus"] == "PIF"].sample(frac=1, random_state=seed)

    def _split_group(group: pd.DataFrame):
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return (
            group.iloc[:n_train],
            group.iloc[n_train : n_train + n_val],
            group.iloc[n_train + n_val :],
        )

    train_def, val_def, test_def = _split_group(defaults)
    train_pif, val_pif, test_pif = _split_group(paid)

    rng = np.random.default_rng(seed)

    def _concat_shuffle(*parts):
        combined = pd.concat(parts).reset_index(drop=True)
        idx = rng.permutation(len(combined))
        return combined.iloc[idx].reset_index(drop=True)

    return (
        _concat_shuffle(train_def, train_pif),
        _concat_shuffle(val_def, val_pif),
        _concat_shuffle(test_def, test_pif),
    )


class SBTransformer:
    """
    Transformer class 
    """

    def __init__(self):
        self._third_party_median = None
        self._col_transformer = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "SBTransformer":
        """Learn transformation statistics from training data."""
        prepped = self._prep(df, fitting=True)
        self._col_transformer = self._build_col_transformer()
        self._col_transformer.fit(prepped[_NUMERIC_COLS + _CAT_COLS])
        self._fitted = True
        return self

    def transform(self, df):
        """
        Transform our sb dataframe

        returns tuple of (features, labels) as np arrays
        """
        prepped = self._prep(df, fitting=False)

        encoded = self._col_transformer.transform(prepped[_NUMERIC_COLS + _CAT_COLS])
        passthrough = prepped[_PASSTHROUGH_COLS].to_numpy(dtype=np.float32)
        binary = prepped[_BINARY_COLS].to_numpy(dtype=np.float32)

        X = np.hstack([encoded, passthrough, binary]).astype(np.float32)
        y = (prepped["loanstatus"] == "CHGOFF").to_numpy(dtype=np.int64)
        return X, y

    def fit_transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Fit on df, then transform it."""
        return self.fit(df).transform(df)

    def get_feature_names(self) -> list[str]:
        """Return feature names matching the columns of X."""
        if not self._fitted:
            raise RuntimeError("Call fit() before get_feature_names().")
        numeric_names = [f"{c}_log_scaled" for c in _NUMERIC_COLS]
        cat_names = list(
            self._col_transformer.named_transformers_["categorical"].get_feature_names_out(_CAT_COLS)
        )
        return numeric_names + cat_names + _PASSTHROUGH_COLS + _BINARY_COLS

    def _prep(self, df: pd.DataFrame, fitting: bool) -> pd.DataFrame:
        """
        Feature engineering applied before the sklearn ColumnTransformer.
        All operations here are either stateless or use statistics stored
        on self (learned during fit).
        """

        if fitting:
            self._third_party_median = df["thirdpartydollars"].median()
        df["thirdpartydollars"] = df["thirdpartydollars"].fillna(self._third_party_median)

        # naics sector is the first two digits of the naics code which we encode
        df["naics_sector"] = (
            df["naicscode"]
            .fillna(0)
            .astype(int)
            .astype(str)
            .str[:2]
        )

        # misc col specific prep
        # Replace any zero thirdpartydollars before division to avoid inf in leverageratio
        safe_third = df["thirdpartydollars"].replace(0, self._third_party_median)
        df["leverageratio"] = df["grossapproval"] / safe_third
        df["approvalyear"] = (df["approvalyear"] - 2010) / 15
        # Unknown term lengths map to NaN — fill with 1 (maps to 240-month equivalent, 0.5 after /2)
        df["terminmonths"] = df["terminmonths"].map({120: 0, 240: 1, 300: 2}).fillna(1) / 2
        df["collateralind"] = (df["collateralind"] == "Y").astype(int)

        return df

    def _build_col_transformer(self):
        """
        Use pipeline and col transformer to easily replicate scaling/encoding steps
        """
        numeric_pipeline = Pipeline([
            ("log", FunctionTransformer(np.log1p, validate=True)),
            ("scale", StandardScaler()),
        ])
        return ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, _NUMERIC_COLS),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                    _CAT_COLS,
                ),
            ],
            remainder="drop",
        )


if __name__ == "__main__":
    data_path = Path(__file__).parent.parent.parent / "data" / "cleaned_business_loans_2010_2025.csv"
    df = pd.read_csv(data_path, low_memory=False)

    train_df, val_df, test_df = stratified_split(df)

    data_dir = data_path.parent

    # Save raw financial values BEFORE fit_transform mutates the dataframes in-place
    train_df[['grossapproval', 'terminmonths']].to_csv(data_dir / "sb_train_raw.csv", index=False)
    val_df[['grossapproval', 'terminmonths']].to_csv(data_dir / "sb_val_raw.csv", index=False)
    test_df[['grossapproval', 'terminmonths']].to_csv(data_dir / "sb_test_raw.csv", index=False)

    transformer = SBTransformer()
    X_train, y_train = transformer.fit_transform(train_df)
    X_val,   y_val   = transformer.transform(val_df)
    X_test,  y_test  = transformer.transform(test_df)

    train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
    val_data = np.hstack([X_val, y_val.reshape(-1, 1)])
    test_data = np.hstack([X_test, y_test.reshape(-1, 1)])

    np.savetxt(data_dir / "sb_train_data.csv", train_data, delimiter=",")
    np.savetxt(data_dir / "sb_val_data.csv", val_data, delimiter=",")
    np.savetxt(data_dir / "sb_test_data.csv", test_data, delimiter=",")
