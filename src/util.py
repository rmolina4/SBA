from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "out"
Style = Enum("Style", ["NORMAL", "PATHS"])
type Panel = tuple[pd.DataFrame, str, Style]
type Panels = tuple[Panel, *tuple[Panel, ...]]


def plot(
    data: pd.DataFrame | Panels,
    fileName: str,
    title: str,
) -> None:
    if isinstance(data, pd.DataFrame):
        panels = ((data, title, Style.NORMAL),)
    else:
        panels = data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(min(9 * len(panels), 15), 6), squeeze=False
    )
    if not isinstance(data, pd.DataFrame):
        fig.suptitle(title)

    for i, (ax, (df, panelTitle, style)) in enumerate(zip(axes[0], panels)):
        options: dict[str, Any]
        match style:
            case Style.NORMAL:
                options = {
                    "style": "o",
                    "markersize": 2,
                }
            case Style.PATHS:
                options = {
                    "color": f"C{i}",
                    "linewidth": 0.5,
                    "alpha": 0.02,
                    "legend": False,
                }
        df.plot(ax=ax, title=panelTitle, **options)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{fileName}.png"), dpi=200)
    plt.close(fig)
