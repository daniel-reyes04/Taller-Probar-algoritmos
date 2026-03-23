import pandas as pd


def categorize_revenue(y):
    labels = ["Bajo (0)", "Medio (1)", "Alto (2)"]
    # qcut distribuye por cuantiles y facilita comparaciones entre clases.
    return pd.qcut(y, q=3, labels=labels, duplicates="drop")
