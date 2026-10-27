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
    history: pd.Series | None = None
    mean: pd.Series | None = None


type Panels = tuple[Panel, *tuple[Panel, ...]]


def plot(
    data: pd.DataFrame | Panels,
    fileName: str,
    title: str,
) -> None:
    if isinstance(data, pd.DataFrame):
        panels = (Panel(data, title, Style.NORMAL),)
    else:
        panels = data

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(min(9 * len(panels), 15), 6), squeeze=False
    )
    if not isinstance(data, pd.DataFrame):
        fig.suptitle(title)

    for i, (ax, panel) in enumerate(zip(axes[0], panels)):
        plotPanel(ax, panel, i)

    fig.tight_layout()
    fig.savefig(str(OUT_DIR / f"{fileName}.png"), dpi=200)
    plt.close(fig)


def plotPanel(ax: Axes, panel: Panel, i: int) -> None:
    df, panelTitle, style, history, mean = panel
    options = STYLE_OPTIONS[style].copy()

    if style is Style.PATHS:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Style.PATHS requires a DataFrame")
        options.update(color=f"C{i}", alpha=len(df.columns) ** -0.5)
        if history is not None:
            start = pd.DataFrame(
                history.iloc[-1], index=history.index[-1:], columns=df.columns
            )
            df = pd.concat([start, df])
            if mean is not None:
                mean = pd.concat([history.iloc[-1:], mean])

    df.plot(ax=ax, title=panelTitle, **options)
    if history is not None:
        history.plot(ax=ax, color=f"C{i}", linewidth=0.5)
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
