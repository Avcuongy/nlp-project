import pandas as pd


def parse_label(label_str):
    clean_str = label_str.replace("{", "").replace("}", "").strip()
    return [tag.strip() for tag in clean_str.split(";") if tag.strip()]
