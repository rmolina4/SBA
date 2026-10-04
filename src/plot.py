import pandas as pd
import matplotlib.pyplot as plt


def plot(df: pd.DataFrame) -> None:
    _, ax = plt.subplots()
    ax.axis("off")
