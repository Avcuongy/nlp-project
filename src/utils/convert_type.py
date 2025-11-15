import pandas as pd

def convert_column_to_numeric(df: pd.DataFrame, col: str, errors: str = "coerce"):
    """
    Convert a single column in a DataFrame to numeric type.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to convert.
        col (str): The name of the column to convert to numeric type.
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaN,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted to numeric type.
    """
    df[col] = pd.to_numeric(df[col], errors=errors)
    return df


def convert_column_to_category(df: pd.DataFrame, col: str):
    """
    Convert a single column in a DataFrame to categorical type.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to convert.
        col (str): The name of the column to convert to categorical type.

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted to categorical type.
    """
    df[col] = df[col].astype("category")
    return df


def convert_column_date(df: pd.DataFrame, col: str, errors: str = "coerce"):
    """
    Convert a single column in a DataFrame to datetime type.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to convert.
        col (str): The name of the column to convert to datetime type.
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaT,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted to datetime type.
    """
    df[col] = pd.to_datetime(df[col], errors=errors)
    return df


def convert_column_date_epoch(df: pd.DataFrame, col: str, errors: str = "coerce"):
    """
    Convert a single column containing epoch timestamps to datetime type.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to convert.
        col (str): The name of the column containing epoch timestamps (in seconds).
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaT,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the specified column converted from epoch to datetime.
    """
    df[col] = pd.to_datetime(df[col], unit="s", errors=errors)
    return df


def convert_column_time_am_pm(
    df: pd.DataFrame, time_col: str, date_col: str = "date", errors: str = "coerce"
):
    """
    Convert a time column with AM/PM format to datetime by combining with a date column.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        time_col (str): The name of the column containing time values with AM/PM format.
        date_col (str, optional): The name of the column containing date values.
                                 Defaults to "date".
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaT,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the time column converted to full datetime.
    """
    df[time_col] = pd.to_datetime(
        df[date_col].dt.strftime("%Y-%m-%d") + " " + df[time_col], errors=errors
    )
    return df


def convert_columns_to_numeric(
    df: pd.DataFrame, cols: list[str], errors: str = "coerce"
):
    """
    Convert multiple columns in a DataFrame to numeric type.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        cols (list[str]): A list of column names to convert to numeric type.
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaN,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the specified columns converted to numeric type.
    """
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors=errors)
    return df


def convert_columns_to_category(df: pd.DataFrame, cols: list[str]):
    """
    Convert multiple columns in a DataFrame to categorical type.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        cols (list[str]): A list of column names to convert to categorical type.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns converted to categorical type.
    """
    for col in cols:
        df[col] = df[col].astype("category")
    return df


def convert_columns_date(df: pd.DataFrame, cols: list[str], errors: str = "coerce"):
    """
    Convert multiple columns in a DataFrame to datetime type.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        cols (list[str]): A list of column names to convert to datetime type.
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaT,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the specified columns converted to datetime type.
    """
    for col in cols:
        df[col] = pd.to_datetime(df[col], errors=errors)
    return df


def convert_columns_date_epoch(
    df: pd.DataFrame, cols: list[str], errors: str = "coerce"
):
    """
    Convert multiple columns containing epoch timestamps to datetime type.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        cols (list[str]): A list of column names containing epoch timestamps (in seconds).
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaT,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".
    """
    for col in cols:
        df[col] = pd.to_datetime(df[col], unit="s", errors=errors)
    return df


def convert_columns_time_am_pm(
    df: pd.DataFrame,
    time_cols: list[str],
    date_col: str = "date",
    errors: str = "coerce",
):
    """
    Convert multiple time columns with AM/PM format to datetime by combining with a date column.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to convert.
        time_cols (list[str]): A list of column names containing time values with AM/PM format.
        date_col (str, optional): The name of the column containing date values.
                                 Defaults to "date".
        errors (str, optional): How to handle conversion errors.
                               'coerce' converts invalid parsing to NaT,
                               'raise' raises an exception,
                               'ignore' returns the original data.
                               Defaults to "coerce".

    Returns:
        pd.DataFrame: The DataFrame with the time columns converted to full datetime.
    """
    for col in time_cols:
        df[col] = pd.to_datetime(
            df[date_col].dt.strftime("%Y-%m-%d") + " " + df[col], errors=errors
        )
    return df
