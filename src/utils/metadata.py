import os
import json
import pandas as pd


def get_default_download_path():
    """
    Gets the default download path based on the operating system.

    Returns:
        The default download path based on the operating system.
    """
    if os.name == "nt":  # Windows
        return os.path.join(os.path.expanduser("~"), "Downloads")
    else:  # macOS, Linux
        return os.path.join(os.path.expanduser("~"), "Downloads")


def save_metadata(df: pd.DataFrame, path: str = None):
    """
    Saves the data types of each column in a DataFrame to a JSON file.

    Args:
        df (pd.DataFrame): The input pandas DataFrame whose column data types are to be saved.
        path (str, optional): The file path where the metadata JSON will be saved.
                             If None, saves to ~/Downloads/metadata.json
    """
    try:
        if path is None:
            path = os.path.join(get_default_download_path(), "metadata.json")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        dtypes = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype = "datetime64[ns]"
            elif pd.api.types.is_categorical_dtype(df[col]):
                dtype = "category"
            dtypes[col] = dtype

        with open(path, "w", encoding="utf-8") as f:
            json.dump(dtypes, f, indent=4)

    except Exception as e:
        raise RuntimeError(f"[save_metadata] Failed: {e}")


def load_metadata(df: pd.DataFrame, path: str = None):
    """
    Loads column data types from a metadata JSON file and casts DataFrame columns.

    Args:
        df (pd.DataFrame): The input pandas DataFrame whose columns are to be casted based on the loaded metadata.
        path (str, optional): The file path to the metadata JSON file.
                             If None, loads from ~/Downloads/metadata.json

    Returns:
        pd.DataFrame: The DataFrame with its columns casted to the data types specified in the metadata file.
    """
    try:
        if path is None:
            path = os.path.join(get_default_download_path(), "metadata.json")

        with open(path, "r", encoding="utf-8") as f:
            dtypes = json.load(f)

        for col, dtype in dtypes.items():
            if col not in df.columns:
                continue
            if dtype.startswith("datetime"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype == "category":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(dtype)

        return df

    except Exception as e:
        raise RuntimeError(f"[load_metadata] Failed: {e}")
