from jaxtyping import install_import_hook

install_import_hook(
    [
        "simulacra.causal",
        "simulacra.covariance",
        "simulacra.family",
        "simulacra.states",
        "simulacra.survival",
        "simulacra.transforms",
    ],
    "beartype.beartype",
)

from .builder import (
    GRAPH,
    DiscreteSurvival,
    PositiveSupportResponse,
    Predictor,
    Response,
    Survival,
    simulate,
)
from .covariance import (
    MaternOrder,
    array_normal,
    compound_symmetry,
    covariance,
    lkj_correlation,
    matern_kernel,
    sample_features,
)
from .family import Family
from .graph import Graph, Transition, build_graph, step
from .pipeline import Pipeline, Run, Step, label
from .states import (
    DiscreteSurvivalData,
    PredictorData,
    RandomEffect,
    ResponseData,
    SurvivalData,
    promote,
)

__all__ = [
    "GRAPH",
    "DiscreteSurvival",
    "DiscreteSurvivalData",
    "Family",
    "Graph",
    "MaternOrder",
    "Pipeline",
    "PositiveSupportResponse",
    "Predictor",
    "PredictorData",
    "RandomEffect",
    "Response",
    "ResponseData",
    "Run",
    "Step",
    "Survival",
    "SurvivalData",
    "Transition",
    "array_normal",
    "build_graph",
    "compound_symmetry",
    "covariance",
    "label",
    "lkj_correlation",
    "matern_kernel",
    "promote",
    "sample_features",
    "simulate",
    "step",
]
