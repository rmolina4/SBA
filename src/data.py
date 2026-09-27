import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"
MORTALITY_BASE_YEAR = 2012
PLOT_DPI = 300


# retiree mortalities
# https://www.soa.org/resources/experience-studies/2019/pri-2012-private-mortality-tables/
def loadMortality() -> pd.DataFrame:
    return pd.read_excel(
        DATA_DIR / "soa_pri2012_amount_weighted_mortality.xlsx",
        sheet_name="Total Dataset",
        header=None,
        index_col=0,
        usecols="B,E,J",
        names=["age", "female", "male"],
        skiprows=5,
    ).dropna()


# mortality improvements
# https://www.soa.org/resources/experience-studies/2021/mortality-improvement-scale-mp-2021/
def loadImprovement() -> pd.DataFrame:
    return (
        pd.concat(
            pd.read_excel(
                DATA_DIR / "soa_mp2021_mortality_improvement.xlsx",
                sheet_name=["Female", "Male"],
                header=1,
                index_col=0,
            ),
            axis=1,
        )
        .rename_axis(index="age", columns=["sex", "year"])
        .rename(columns=str.lower, level="sex")
        .rename(
            columns=lambda year: (
                int(year.removesuffix("+")) if isinstance(year, str) else year
            ),
            level="year",
        )
        .rename(index=lambda age: 20 if age == "≤ 20" else int(age))
    )


def loadCurve(pattern: str) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_excel(file, index_col=0, header=[3, 4])
            .dropna(axis=0, how="all")
            .dropna(axis=1, how="all")
            for file in DATA_DIR.glob(pattern)
        ],
        axis=1,
        verify_integrity=True,
    )

    return (
        df.set_axis(
            pd.to_datetime(
                [f"{y}-{m}" for y, m in df.columns],
                format="%Y-%b",
            ),
            axis=1,
        )
        .T.rename_axis(index="date", columns="maturity")
        .sort_index()
    )


# Treasury spot curves
# https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves/treasury-coupon-issues
def loadTreasury() -> pd.DataFrame:
    return loadCurve("tnc_treasury_spot_curve_*.xls*")


# Corporate bond spot curves
# https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curve/corporate-bond-yield-curve
def loadCorporate() -> pd.DataFrame:
    return loadCurve("hqm_corporate_bond_spot_curve_*.xls*")


def loadSurvival() -> pd.DataFrame:
    mortality = loadMortality()
    improvement = loadImprovement()
    mortality = pd.concat(
        {
            sex: pd.concat(
                [
                    mortality[sex].rename(MORTALITY_BASE_YEAR),
                    1
                    - improvement[sex].loc[mortality.index, MORTALITY_BASE_YEAR + 1 :],
                ],
                axis=1,
            ).cumprod(axis=1)
            for sex in mortality.columns
        },
        axis=1,
        names=["sex", "year"],
    )
    return (1 - mortality).shift(1, fill_value=1).cumprod()


def plotField(data: pd.DataFrame, title: str, field: str) -> tuple[Figure, list[Axes]]:
    groups = data.columns.get_level_values(field).unique()
    years = data.columns.get_level_values("year").unique().sort_values()
    norm = Normalize(vmin=years.min(), vmax=years.max())
    cmap = plt.get_cmap("winter")
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(12, 6),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel().tolist()
    for ax, group in zip(axes, groups):
        data.xs(group, level=field, axis=1).loc[:, years].plot(
            ax=ax,
            color=cmap(norm(years)),
            legend=False,
        )
        ax.set_title(str(group).capitalize())
    fig.suptitle(title)
    fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        ticks=years[::5].union(years[-1:]),
    )
    return fig, axes


def plot(
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    field: str | None = None,
    exportFormat: str = "png",
) -> None:
    if field is not None:
        fig, axes = plotField(data, title, field)
    else:
        ax = data.plot(figsize=(10, 6), marker="o", linestyle="None")
        ax.get_legend().set_title(None)
        ax.set_title(title)
        fig, axes = ax.figure, [ax]

    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(ylabel)
    if field is None:
        fig.tight_layout()
    OUT_DIR.mkdir(exist_ok=True)
    fig.savefig(
        (OUT_DIR / filename).with_suffix(f".{exportFormat}"),
        format=exportFormat,
        dpi=PLOT_DPI,
        bbox_inches="tight",
    )


def exportLatex(
    vector: pd.Series | pd.DataFrame,
    filename: str,
    rowCount: int = 3,
    columnCount: int = 3,
) -> None:
    matrix = vector.to_frame() if isinstance(vector, pd.Series) else vector
    rowIndices, columnIndices = [
        (
            list(range(size))
            if size <= 2 * count
            else list(range(size)[:count]) + [None] + list(range(size)[size - count :])
        )
        for size, count in zip(matrix.shape, (rowCount, columnCount))
    ]
    entries = (
        " & ".join(
            (
                (r"\ddots" if columnIndex is None else r"\vdots")
                if rowIndex is None
                else (
                    r"\cdots"
                    if columnIndex is None
                    else f"{matrix.iloc[rowIndex, columnIndex]:.2f}"
                )
            )
            for columnIndex in columnIndices
        )
        for rowIndex in rowIndices
    )
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / filename).write_text(
        "$$\n\\begin{vmatrix}\n" + " \\\\\n".join(entries) + "\n\\end{vmatrix}\n$$\n",
        encoding="utf-8",
    )


def adfTest(data: pd.DataFrame) -> pd.Series:
    return data.apply(
        lambda col: adfuller(col.dropna(), result_object=True).pvalue
    ).rename("pvalue")


if __name__ == "__main__":
    # print(loadMortality())
    # print(loadImprovement())
    # print(loadSurvival())
    print(loadCorporate())
