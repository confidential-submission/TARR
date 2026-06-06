"""
Dataset loaders for Adult (Census Income), German Credit, and Bank Marketing
(paper Section 4.1).

Preprocessing follows the paper:
- Categorical features: one-hot encoded.
- Continuous features: normalised to [0, 1].
- Sensitive attribute: binarised and appended as the last column of X so that
  sensitive_idx = n_features - 1 is always consistent.

Splits: 70 / 15 / 15  (train / val / test), stratified by label.

Datasets are fetched via scikit-learn's fetch_openml, which caches locally
after the first download.
"""

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from torch.utils.data import TensorDataset


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(-1),
    )


def _three_way_split(X, y, val_frac, test_frac, seed):
    """70 / val_frac / test_frac stratified split."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=y,
    )
    ratio = test_frac / (val_frac + test_frac)
    X_vl, X_te, y_vl, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=ratio,
        random_state=seed,
        stratify=y_tmp,
    )
    return X_tr, X_vl, X_te, y_tr, y_vl, y_te


def _preprocess(df, y: np.ndarray, s: np.ndarray,
                val_frac: float, test_frac: float, seed: int):
    """
    One-hot encode categoricals, MinMax-scale continuous features,
    then append the binary sensitive attribute s as the last column.

    Returns: (train_ds, val_ds, test_ds, n_features, sensitive_idx)
    """
    cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
    num_cols = df.select_dtypes(exclude=['category', 'object']).columns.tolist()

    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

    ct = ColumnTransformer(
        [('ohe', ohe, cat_cols), ('scl', MinMaxScaler(), num_cols)],
        remainder='drop',
    )
    X_body = ct.fit_transform(df).astype(np.float32)
    X = np.hstack([X_body, s.reshape(-1, 1).astype(np.float32)])
    sensitive_idx = X.shape[1] - 1

    X_tr, X_vl, X_te, y_tr, y_vl, y_te = _three_way_split(
        X, y, val_frac, test_frac, seed
    )
    return (
        _to_dataset(X_tr, y_tr),
        _to_dataset(X_vl, y_vl),
        _to_dataset(X_te, y_te),
        X.shape[1],       # n_features (includes the sensitive column)
        sensitive_idx,
    )


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_adult(
    sensitive: str = 'sex',
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Adult (Census Income) dataset via OpenML id 1590.

    sensitive ∈ {'sex', 'race', 'age'}
        sex  : Male → 1, Female → 0
        race : White → 1, non-White → 0
        age  : age ≥ 40 → 1, else 0  (binarisation threshold from [26, 40])

    Returns: (train_ds, val_ds, test_ds, n_features, sensitive_idx)
    """
    ds = fetch_openml(data_id=1590, as_frame=True, parser='auto')
    df = ds.frame.copy()

    y = df['class'].str.strip().str.rstrip('.').isin(['>50K']).astype(float).values
    df.drop(columns=['class', 'fnlwgt'], inplace=True)

    if sensitive == 'sex':
        s = (df['sex'].str.strip() == 'Male').astype(float).values
        df.drop(columns=['sex'], inplace=True)
    elif sensitive == 'race':
        s = (df['race'].str.strip() == 'White').astype(float).values
        df.drop(columns=['race'], inplace=True)
    elif sensitive == 'age':
        s = (df['age'].astype(float) >= 40).astype(float).values
        df.drop(columns=['age'], inplace=True)
    else:
        raise ValueError(f"Adult: unknown sensitive attribute '{sensitive}'. "
                         "Choose from 'sex', 'race', 'age'.")

    return _preprocess(df, y, s, val_frac, test_frac, seed)


def load_credit(
    sensitive: str = 'age',
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    German Credit (credit-g) dataset via OpenML id 31.

    sensitive ∈ {'age', 'sex'}
        age : age ≥ 25 → 1, else 0
        sex : male personal_status → 1, female → 0

    Returns: (train_ds, val_ds, test_ds, n_features, sensitive_idx)
    """
    ds = fetch_openml(data_id=31, as_frame=True, parser='auto')
    df = ds.frame.copy()

    y = (df['class'] == 'good').astype(float).values
    df.drop(columns=['class'], inplace=True)

    if sensitive == 'age':
        s = (df['age'].astype(float) >= 25).astype(float).values
        df.drop(columns=['age'], inplace=True)
    elif sensitive == 'sex':
        # personal_status encodes sex: values start with 'male' or 'female'
        s = df['personal_status'].str.startswith('male').astype(float).values
        df.drop(columns=['personal_status'], inplace=True)
    else:
        raise ValueError(f"Credit: unknown sensitive attribute '{sensitive}'. "
                         "Choose from 'age', 'sex'.")

    return _preprocess(df, y, s, val_frac, test_frac, seed)


def load_bank(
    sensitive: str = 'age',
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
):
    """
    Bank Marketing dataset via OpenML id 1461.

    sensitive ∈ {'age'}
        age : age ≥ 50 → 1, else 0  (binarisation threshold from [26, 40])

    Returns: (train_ds, val_ds, test_ds, n_features, sensitive_idx)
    """
    ds = fetch_openml(data_id=1461, as_frame=True, parser='auto')
    df = ds.frame.copy()

    # OpenML 1461 encodes target as V17 with values '1' / '2'
    label_col = 'V17' if 'V17' in df.columns else df.columns[-1]
    y = (df[label_col].astype(str).str.strip() == '2').astype(float).values
    df.drop(columns=[label_col], inplace=True)

    if sensitive == 'age':
        age_col = 'V1' if 'V1' in df.columns else 'age'
        s = (df[age_col].astype(float) >= 50).astype(float).values
        df.drop(columns=[age_col], inplace=True)
    else:
        raise ValueError(f"Bank: unknown sensitive attribute '{sensitive}'. "
                         "Choose 'age'.")

    return _preprocess(df, y, s, val_frac, test_frac, seed)
