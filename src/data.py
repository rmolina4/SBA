from pathlib import Path

import pandas as pd
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = Path(__file__).resolve().parent.parent / "out"

MORTALITY_BASE_YEAR = 2012


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
            columns=lambda year: 2037 if year == "2037+" else int(year), level="year"
        )
        .rename(index=lambda age: 20 if age == "≤ 20" else int(age))
    )


def loadSurvival() -> pd.DataFrame:
    mortality = loadMortality()
    improvement = loadImprovement().loc[
        mortality.index, pd.IndexSlice[:, MORTALITY_BASE_YEAR + 1 :]
    ]
    mortality = pd.concat(
        {
            sex: pd.concat(
                [
                    mortality[sex].rename(MORTALITY_BASE_YEAR),
                    1 - improvement[sex],
                ],
                axis=1,
            ).cumprod(axis=1)
            for sex in mortality
        },
        axis=1,
        names=["sex", "year"],
    )
    return 1 - mortality


def loadRetireeSurvival(sex: str, birth: int, start: int) -> pd.DataFrame:
    df = (
        loadSurvival()
        .loc[start - birth :, pd.IndexSlice[sex, start:]]
        .to_numpy()
        .diagonal()
    )
    steps = np.arange(len(df) + 1)
    return pd.DataFrame(
        {
            "year": start + steps,
            "survival": np.r_[1.0, df].cumprod(),
        },
        index=pd.Index(start - birth + steps, name="age"),
    )


def loadCurve(pattern: str) -> pd.DataFrame:
    df = pd.concat(
        [
            pd.read_excel(file, index_col=0, header=[3, 4])
            .dropna(how="all")
            .dropna(axis=1, how="all")
            for file in DATA_DIR.glob(pattern)
        ],
        axis=1,
        verify_integrity=True,
    )
    df.columns = pd.to_datetime(
        [f"{year}-{month}" for year, month in df.columns], format="%Y-%b"
    )
    return df.T.rename_axis(index="date", columns="maturity").sort_index()


# treasury spot curves
# https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curves/treasury-coupon-issues
def loadTreasury() -> pd.DataFrame:
    return loadCurve("tnc_treasury_spot_curve_*.xls*")


# corporate bond spot curves
# https://home.treasury.gov/data/treasury-coupon-issues-and-corporate-bond-yield-curve/corporate-bond-yield-curve
def loadCorporate() -> pd.DataFrame:
    return loadCurve("hqm_corporate_bond_spot_curve_*.xls*")


def loadSpread() -> pd.DataFrame:
    treasury = loadTreasury()
    corporate = loadCorporate()
    return (corporate - treasury).dropna()


# synthetic benefits
# https://www.ssa.gov/policy/docs/microdata/bepuf-2020/index.html
def loadPension() -> pd.DataFrame:
    return (
        pd.read_csv(
            DATA_DIR / "BEPUF_2020_benefits.csv",
            index_col="ID",
        )
        .query("IP == 'R' and BT == 'A'")
        .rename(columns=str.lower)
    )


if __name__ == "__main__":
    survival = loadSurvival()
    treasury = loadTreasury()
    spread = loadSpread()
    pension = loadPension()
    retireeSurvival = loadRetireeSurvival("female", 1955, 2022)
    print(survival)
    print(treasury)
    print(spread)
    print(pension)
    print(f'${pension["mbc"].sum():,.2f}')
    print(retireeSurvival)
