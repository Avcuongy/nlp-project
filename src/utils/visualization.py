import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

sns.set_style("whitegrid")


def plot_multiple_histograms(
    df: pd.DataFrame,
    cols: List[str],
    axis: int = 0,
    hue: Optional[str] = None,
    multiple: str = "layer",
    kde: bool = False,
    common_norm: bool = True,
    color: Optional[str] = None,
    palette: Optional[str] = None,
    bins: int = 30,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Creates multiple histograms in a subplot grid for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        cols (List[str]): List of numerical column names to plot.
        axis (int): 0 for horizontal (x-axis), 1 for vertical (y-axis).
        hue (Optional[str]): Column name for categorical grouping.
        multiple (str): How to represent multiple distributions ('layer', 'stack', 'dodge', 'fill').
        kde (bool): If True, overlays a KDE curve on each histogram.
        common_norm (bool): Normalize density across groups if True.
        color (Optional[str]): Single color for histograms when hue is None.
        palette (Optional[str]): Color palette for hue groups.
        bins (int): Number of histogram bins.
        n_cols (int): Number of columns in the subplot grid.
        figsize (Optional[Tuple[int, int]]): Figure size as (width, height).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not cols:
        raise ValueError("List of columns must not be empty.")
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (horizontal/x) or 1 (vertical/y).")

    axis_str = "x" if axis == 0 else "y"

    n_plots = len(cols)
    n_rows = math.ceil(n_plots / n_cols)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.histplot(
            data=df,
            **{axis_str: col},
            hue=hue,
            multiple=multiple,
            bins=bins,
            kde=kde,
            common_norm=common_norm,
            palette=palette,
            color=color,
            ax=axes[i],
        )
        axes[i].set_title(f"Histogram: {col}")
        if axis == 0:
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Frequency")
        else:
            axes[i].set_xlabel("Frequency")
            axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_multiple_kdeplots(
    df: pd.DataFrame,
    cols: List[str],
    axis: int = 0,
    hue: Optional[str] = None,
    fill: Optional[bool] = None,
    common_norm: bool = True,
    color: Optional[str] = None,
    palette: Optional[str] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Creates multiple KDE plots in a subplot grid for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        cols (List[str]): List of numerical column names to plot.
        axis (int): 0 for horizontal (x-axis), 1 for vertical (y-axis).
        hue (Optional[str]): Column name for categorical grouping.
        fill (Optional[bool]): Whether to fill the area under the KDE curve.
        common_norm (bool): Normalize density across groups if True.
        color (Optional[str]): Single color for KDE plots when hue is None.
        palette (Optional[str]): Color palette for hue groups.
        n_cols (int): Number of columns in the subplot grid.
        figsize (Optional[Tuple[int, int]]): Figure size as (width, height).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not cols:
        raise ValueError("List of columns must not be empty.")
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (horizontal/x) or 1 (vertical/y).")

    axis_str = "x" if axis == 0 else "y"

    n_plots = len(cols)
    n_rows = math.ceil(n_plots / n_cols)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.kdeplot(
            data=df,
            **{axis_str: col},
            hue=hue,
            palette=palette,
            color=color,
            ax=axes[i],
            fill=fill,
            common_norm=common_norm,
        )
        axes[i].set_title(f"KDE: {col}")
        if axis == 0:
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Density")
        else:
            axes[i].set_xlabel("Density")
            axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()


def plot_multiple_boxplots(
    df: pd.DataFrame,
    cols: List[str],
    axis: int = 0,
    hue: Optional[str] = None,
    color: Optional[str] = None,
    palette: Optional[str] = None,
    n_cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Creates multiple boxplots in a subplot grid for numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        cols (List[str]): List of numerical column names to plot.
        axis (int): 0 for horizontal (x-axis), 1 for vertical (y-axis).
        hue (Optional[str]): Column name for categorical grouping.
        color (Optional[str]): Single color for boxplots when hue is None.
        palette (Optional[str]): Color palette for hue groups.
        n_cols (int): Number of columns in the subplot grid.
        figsize (Optional[Tuple[int, int]]): Figure size as (width, height).
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not cols:
        raise ValueError("List of columns must not be empty.")
    if axis not in [0, 1]:
        raise ValueError("axis must be 0 (horizontal/x) or 1 (vertical/y).")

    axis_str = "x" if axis == 0 else "y"

    n_plots = len(cols)
    n_rows = math.ceil(n_plots / n_cols)
    if figsize is None:
        figsize = (n_cols * 5, n_rows * 4)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(cols):
        sns.boxplot(
            data=df,
            **{axis_str: col},
            hue=hue,
            palette=palette,
            color=color,
            ax=axes[i],
        )
        axes[i].set_title(f"Boxplot: {col}")
        if axis == 0:
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(hue if hue else "")
        else:
            axes[i].set_xlabel(hue if hue else "")
            axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()
