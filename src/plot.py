from typing import Any, NamedTuple
from enum import Enum
from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "out"
Style = Enum("Style", ["NORMAL", "PATHS"])
STYLE_OPTIONS: dict[Style, dict[str, Any]] = {
    Style.NORMAL: {"style": "o", "markersize": 2, "legend": True},
    Style.PATHS: {"linewidth": 0.5, "legend": False},
}


class Panel(NamedTuple):
    data: pd.DataFrame | pd.Series
    title: str
    style: Style
    mean: pd.Series | None = None


type Panels = tuple[Panel, *tuple[Panel, ...]]


def plot(
    data: pd.DataFrame | Panels,
    fileName: str,
    title: str,
) -> None:
    showFigureTitle = not isinstance(data, pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        data = (Panel(data, title, Style.NORMAL),)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(data), figsize=(min(9 * len(data), 15), 6), squeeze=False
    )
    if showFigureTitle:
        fig.suptitle(title)

    for i, (ax, panel) in enumerate(zip(axes[0], data)):
        plotPanel(ax, panel, i)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{fileName}.png"), dpi=200)
    plt.close(fig)


def plotPanel(ax: Axes, panel: Panel, i: int) -> None:
    df, panelTitle, style, mean = panel
    options = STYLE_OPTIONS[style].copy()

    if style is Style.PATHS:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Style.PATHS requires a DataFrame")
        options.update(color=f"C{i}", alpha=len(df.columns) ** -0.5)

    df.plot(ax=ax, title=panelTitle, **options)
    if mean is not None:
        color = tuple(channel * 0.6 for channel in to_rgb(f"C{i}"))
        mean.plot(
            ax=ax,
            color=color,
            linestyle="--",
            linewidth=2,
            legend=True,
            label="Mean",
        )
