from typing import Any, NamedTuple
from enum import Enum
from pathlib import Path
from matplotlib.axes import Axes

import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent / "out"
Style = Enum("Style", ["POINTS", "LINE", "DASHED", "PATHS"])
STYLE_OPTIONS: dict[Style, dict[str, Any]] = {
    Style.POINTS: {"style": "o", "markersize": 2, "legend": True},
    Style.LINE: {"linestyle": "-", "linewidth": 1.5, "legend": True},
    Style.DASHED: {"linestyle": "--", "linewidth": 2, "legend": True},
    Style.PATHS: {"linewidth": 0.5, "legend": False},
}


class Layer(NamedTuple):
    data: pd.DataFrame | pd.Series
    style: Style
    color: str | None = None


type Layers = tuple[Layer, *tuple[Layer, ...]]


class Panel(NamedTuple):
    title: str
    layers: Layers


type Panels = tuple[Panel, *tuple[Panel, ...]]


def plot(
    data: pd.DataFrame | Panels,
    fileName: str,
    title: str,
) -> None:
    showFigureTitle = not isinstance(data, pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        data = (Panel(title, (Layer(data, Style.POINTS),)),)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(data), figsize=(min(9 * len(data), 15), 6), squeeze=False
    )
    if showFigureTitle:
        fig.suptitle(title)
    for ax, panel in zip(axes[0], data):
        plotPanel(ax, panel)
    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{fileName}.png"), dpi=200)
    plt.close(fig)


def plotPanel(ax: Axes, panel: Panel) -> None:
    ax.set_title(panel.title)
    for df, style, color in panel.layers:
        options = STYLE_OPTIONS[style].copy()
        if style is Style.PATHS:
            options["alpha"] = len(df.columns) ** -0.5
        if color is not None:
            options["color"] = color
        df.plot(ax=ax, **options)
