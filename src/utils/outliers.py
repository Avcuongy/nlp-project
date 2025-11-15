import pandas as pd
import numpy as np
from scipy import stats

def detect_outliers_by_iqr(df: pd.DataFrame, cols: list[str]) -> dict:
    """
    Detects outliers in specified columns of a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze.
        cols (list[str]): List of column names to check for outliers.

    Returns:
        dict: Dictionary mapping column names to lists of outlier indices.
    """
    df_copy = df.copy()
    outliers_dict = {}
    for col in cols:
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df_copy[
            (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
        ].index.tolist()
        outliers_dict[col] = outliers
    return outliers_dict


def detect_outliers_by_zscore(
    df: pd.DataFrame, cols: list[str], threshold: float = 3.0
) -> dict:
    """
    Detects outliers in specified columns using the Z-score method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (list[str]): List of column names to check for outliers.
        threshold (float, optional): Z-score threshold. Defaults to 3.0.

    Returns:
        dict: Dictionary mapping column names to lists of outlier indices.
    """
    df_copy = df.copy()
    outliers_dict = {}
    for col in cols:
        z_scores = np.abs(stats.zscore(df_copy[col], nan_policy="omit"))
        outliers = df_copy[z_scores > threshold].index.tolist()
        outliers_dict[col] = outliers
    return outliers_dict


def cap_outliers_by_iqr(
    df: pd.DataFrame, col: str, lower: float = None, upper: float = None
) -> pd.DataFrame:
    """
    Caps outliers from a single column in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to process.
        lower (float, optional): Lower bound for clipping. If None, calculated via IQR.
        upper (float, optional): Upper bound for clipping. If None, calculated via IQR.

    Returns:
        pd.DataFrame: A copy of the DataFrame with outliers in the specified column capped.
    """
    df_clean = df.copy()

    if lower is not None and upper is not None:
        df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
    else:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)

    return df_clean


def remove_outliers_by_iqr(
    df: pd.DataFrame, col: str, lower: float = None, upper: float = None
) -> pd.DataFrame:
    """
    Removes outliers from a single column in a DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame to clean.
        col (str): Name of the column to remove outliers from.

    Returns:
        pd.DataFrame: DataFrame with outliers removed from the specified column.
    """
    df_clean = df.copy()

    if lower is not None and upper is not None:
        df_clean[col] = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    else:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_clean = df_clean[
            (df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)
        ]

    return df_clean


def log_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Applies log1p transformation to specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to transform.
        cols (list[str]): List of column names to apply log1p transformation.

    Returns:
        pd.DataFrame: DataFrame with transformed columns using log1p.
    """
    df_transformed = df.copy()
    for col in cols:
        df_transformed[col] = np.log1p(df_transformed[col])
    return df_transformed


def sqrt_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Applies square root transformation to specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to transform.
        cols (list[str]): List of column names to apply square root transformation.

    Returns:
        pd.DataFrame: DataFrame with transformed columns using square root.
    """
    df_transformed = df.copy()
    for col in cols:
        df_transformed[col] = np.sqrt(df_transformed[col].clip(lower=1e-10))
    return df_transformed


def boxcox_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Applies Box-Cox transformation to specified columns of a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to transform.
        cols (list[str]): List of column names to apply Box-Cox transformation.

    Returns:
        pd.DataFrame: DataFrame with transformed columns using Box-Cox.
    """
    df_transformed = df.copy()
    for col in cols:
        df_transformed[col] = df_transformed[col].clip(lower=1e-10)
        df_transformed[col], _ = stats.boxcox(df_transformed[col])
    return df_transformed
