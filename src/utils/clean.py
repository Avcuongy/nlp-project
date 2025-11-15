import numpy as np
import pandas as pd
from dateutil.parser import parse
from datetime import datetime


def group_columns_by_type(df: pd.DataFrame, display_info: bool = False):
    """
    Groups DataFrame columns by data type into numerical, categorical, and datetime categories.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze.
        display_info (bool, optional): If True, prints the count and names of columns for each type. Defaults to False.

    Returns:
        tuple: Lists of numerical, categorical, and datetime column names.
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist() or []
    categorical_cols = (
        df.select_dtypes(
            exclude=[np.number, "datetime64[ns]", "datetime", "datetime64"]
        ).columns.tolist()
        or []
    )
    date_cols = (
        df.select_dtypes(
            include=["datetime64[ns]", "datetime", "datetime64"]
        ).columns.tolist()
        or []
    )

    if display_info:
        if numerical_cols:
            print(f"Total numeric columns: {len(numerical_cols)}")
            print("Numeric columns:", numerical_cols)
            print()

        if categorical_cols:
            print(f"Total categorical columns: {len(categorical_cols)}")
            print("Categorical columns:", categorical_cols)
            print()

        if date_cols:
            print(f"Total datetime columns: {len(date_cols)}")
            print("Datetime columns:", date_cols)
            print()

    return numerical_cols, categorical_cols, date_cols


def summary_column_groups(
    numeric_cols: list[str] = [],
    categorical_cols: list[str] = [],
    date_cols: list[str] = [],
):
    """
    Prints a summary of column groups by type.

    Args:
        numeric_cols (list[str], optional): List of numerical column names. Defaults to [].
        categorical_cols (list[str], optional): List of categorical column names. Defaults to [].
        date_cols (list[str], optional): List of datetime column names. Defaults to [].
    """
    if numeric_cols:
        print(f"Total numeric columns: {len(numeric_cols)}")
        print("Numeric columns:", numeric_cols)
        print()

    if categorical_cols:
        print(f"Total categorical columns: {len(categorical_cols)}")
        print("Categorical columns:", categorical_cols)
        print()

    if date_cols:
        print(f"Total datetime columns: {len(date_cols)}")
        print("Datetime columns:", date_cols)
        print()


def detect_columns_datetime(
    df: pd.DataFrame,
    ratio: float = 0.05,
    threshold: float = 0.5,
) -> list:
    """
    Detect all columns that are datetime-like (full date or month+year):
    - Columns already with dtype datetime64
    - Object/string columns that mostly contain valid full datetime values
      (must have at least month+year or day+month+year)

    Args:
        df (pd.DataFrame): DataFrame input
        ratio (float): fraction of non-null values to sample
        threshold (float): minimum success ratio to classify as datetime

    Returns:
        list: column names detected as datetime
    """
    datetime_cols = []

    valid_formats = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%m/%Y",
        "%Y/%m",
    ]

    for col in df.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns:
        datetime_cols.append(col)

    candidate_cols = df.select_dtypes(exclude=["datetime64", "datetime64[ns]"]).columns

    for col in candidate_cols:
        series = df[col].dropna()
        if series.empty:
            continue

        sample_size = max(10, int(len(series) * ratio))
        sample_size = min(sample_size, len(series))
        sample = series.sample(n=sample_size, random_state=42)

        def matches_datetime_format(x):
            if not isinstance(x, str):
                return False
            for fmt in valid_formats:
                try:
                    dt = datetime.strptime(x.strip(), fmt)
                    return True
                except ValueError:
                    continue
            return False

        success_rate = sample.apply(matches_datetime_format).mean()

        if success_rate >= threshold:
            datetime_cols.append(col)

    return datetime_cols
