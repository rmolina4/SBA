from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "out"


def plot(
    df: pd.DataFrame,
    title: str,
    fileName: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.plot(title=title, figsize=(9, 6), style="o", markersize=2)
    plt.savefig(OUT_DIR / f"{fileName}.png", dpi=200)
