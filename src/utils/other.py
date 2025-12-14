import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def parse_label(label_str: str) -> list[str]:
    """Parse label string of format '{label1; label2; ...}' to list of labels.

    Args:
        label_str: String containing labels in curly braces separated by semicolons.

    Returns:
        List of individual label strings.
    """
    clean_str = label_str.replace("{", "").replace("}", "").strip()
    return [tag.strip() for tag in clean_str.split(";") if tag.strip()]


def matrix_labels(df_label: pd.DataFrame) -> tuple[pd.DataFrame, MultiLabelBinarizer]:
    """Convert label strings in DataFrame to a binary matrix representation.
    Args:
        df_label: DataFrame with a column 'labels' containing label strings.
    Returns:
        A tuple containing:
            - DataFrame with binary matrix representation of labels.
            - Fitted MultiLabelBinarizer instance.
    """
    mlb = MultiLabelBinarizer()
    df_label["parsed_label"] = df_label["label"].apply(parse_label)
    matrix_label = mlb.fit_transform(df_label["parsed_label"])
    matrix_label_df = pd.DataFrame(matrix_label, columns=mlb.classes_)
    return matrix_label_df, mlb


if __name__ == "__main__":
    data = {"label": ["{cat; dog}", "{dog; mouse}", "{cat; mouse; rabbit}"]}
    df = pd.DataFrame(data)
    matrix_df, mlb = matrix_labels(df)
    print(matrix_df)
    print("Classes:", mlb.classes_)
