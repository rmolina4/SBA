import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


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


def plot(
    data: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
) -> None:
    ax = data.plot(figsize=(10, 6), marker="o", linestyle="None")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.figure.tight_layout()
    FIGURES_DIR.mkdir(exist_ok=True)
    ax.figure.savefig(FIGURES_DIR / filename, bbox_inches="tight")


def adfTest(data: pd.DataFrame) -> pd.Series:
    return data.apply(
        lambda col: adfuller(col.dropna(), result_object=True).pvalue
    ).rename("pvalue")


if __name__ == "__main__":
    print(loadMortality())
    print(loadImprovement())
    print(loadTreasury())
    print(loadCorporate())
