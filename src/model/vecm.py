from statsmodels.tsa.vector_ar.vecm import VECMResults

from .esg import ESG
from .diagnostic import VECM_RULES, Result, Rule


class ESGVecm(ESG[VECMResults]):
    __slots__ = ()

    def getRules(self) -> tuple[Rule[Result] | Rule[VECMResults], ...]:
        return VECM_RULES
