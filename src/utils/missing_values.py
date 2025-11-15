import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def missing_rate_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Summarizes missing data count and percentage for specified columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze.
        cols (list[str]): List of column names to check for missing values.

    Returns:
        pd.DataFrame: DataFrame with columns 'Missing Count' and 'Missing Rate (%)', sorted by missing rate in descending order.
    """
    total = df.shape[0]
    missing_count = df[cols].isna().sum()
    missing_ratio = (missing_count / total * 100).round(2)

    summary = pd.DataFrame(
        {"Missing Count": missing_count, "Missing Rate (%)": missing_ratio}
    ).sort_values(by="Missing Rate (%)", ascending=False)

    if missing_count.sum() == 0:
        print("No columns have missing values")

    return summary


def plot_missing_rate(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Visualizes the missing data percentage for specified columns using a bar plot.

    Args:
        df (pd.DataFrame): Input DataFrame to analyze.
        cols (list[str]): List of column names to check for missing values.
    """
    total = df.shape[0]
    missing_count = df[cols].isna().sum()
    missing_ratio = (missing_count / total * 100).round(2)

    summary = pd.DataFrame(
        {"Column": cols, "Missing Rate (%)": missing_ratio}
    ).sort_values(by="Missing Rate (%)", ascending=False)

    plt.figure(figsize=(16, 5))
    sns.barplot(x="Column", y="Missing Rate (%)", data=summary, palette="coolwarm")
    plt.title("Missing Rate by Column", fontsize=14, pad=10)
    plt.xlabel("Columns", fontsize=12)
    plt.ylabel("Missing Rate (%)", fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
