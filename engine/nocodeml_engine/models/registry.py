"""Model registry.

Models are described by metadata, not hard-coded across the app. The UI, the
validator, and the trainer all read from here, so adding a model means adding
one entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from nocodeml_engine.config import ModelConfig, TaskType

MAX_MODELS = 5


@dataclass(frozen=True)
class HyperParam:
    name: str
    kind: str  # "int" | "float" | "choice" | "bool" | "optional_int"
    default: Any
    choices: tuple = ()
    min: float | None = None
    max: float | None = None
    advanced: bool = False
    description: str = ""


@dataclass(frozen=True)
class RegularizationSpec:
    """Which regularization a model supports.

    kind "penalty": a norm penalty chosen from `options` (strength via `strength`).
    kind "complexity": no L1/L2; capacity is limited via listed hyperparameters.
    kind "none": nothing meaningful to expose.
    """

    kind: str
    options: tuple[str, ...] = ()
    complexity_params: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    estimators: dict[TaskType, Callable[..., Any]]
    requires_scaling: bool
    hyperparameters: tuple[HyperParam, ...]
    regularization: dict[TaskType, RegularizationSpec]
    cost: str  # "low" | "medium" | "high"
    interpretability: str  # "high" | "medium" | "low"
    recommend: Callable[[int, int], dict[str, Any]] = field(
        default=lambda n_rows, n_features: {}
    )

    @property
    def tasks(self) -> list[TaskType]:
        return list(self.estimators)


class ModelConfigError(ValueError):
    pass


# --- helpers translating "regularization" config into estimator params ----


def _logreg_params(reg: dict[str, Any]) -> dict[str, Any]:
    kind = reg.get("type", "l2")
    out: dict[str, Any] = {}
    if kind == "none":
        out["C"] = math.inf
    else:
        out["C"] = float(reg.get("strength", 1.0))
        out["l1_ratio"] = {"l1": 1.0, "l2": 0.0}.get(kind, float(reg.get("l1_ratio", 0.5)))
    # lbfgs cannot do an L1 component; saga can.
    out["solver"] = "saga" if kind in ("l1", "elasticnet") else "lbfgs"
    return out


def _alpha_params(reg: dict[str, Any]) -> dict[str, Any]:
    return {"alpha": float(reg["strength"])} if "strength" in reg else {}


def _svm_params(reg: dict[str, Any]) -> dict[str, Any]:
    return {"C": float(reg["strength"])} if "strength" in reg else {}


_PENALTY_TRANSLATORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "logistic_regression": _logreg_params,
    "ridge": _alpha_params,
    "lasso": _alpha_params,
    "svm": _svm_params,
}

_TREE_COMPLEXITY = ("max_depth", "min_samples_split", "min_samples_leaf", "max_features")

_TREE_REG = RegularizationSpec(
    kind="complexity",
    complexity_params=_TREE_COMPLEXITY,
    note="Tree models are regularized by limiting their size, not by L1/L2.",
)


def _tree_hparams(extra: tuple[HyperParam, ...] = ()) -> tuple[HyperParam, ...]:
    return (
        HyperParam("max_depth", "optional_int", None, min=1, max=100,
                   description="Maximum tree depth. None grows until leaves are pure."),
        HyperParam("min_samples_split", "int", 2, min=2, max=1000, advanced=True),
        HyperParam("min_samples_leaf", "int", 1, min=1, max=1000, advanced=True),
        HyperParam("max_features", "choice", None, choices=(None, "sqrt", "log2"), advanced=True),
        *extra,
    )


_FOREST_EXTRA = (
    HyperParam("n_estimators", "int", 100, min=10, max=1000, description="Number of trees."),
    HyperParam("bootstrap", "bool", True, advanced=True),
)


def _recommend_forest(n_rows: int, n_features: int) -> dict[str, Any]:
    # Small data: shallower trees to curb overfitting; large data: cap cost.
    return {
        "n_estimators": 100 if n_rows < 50_000 else 200,
        "max_depth": 10 if n_rows < 1_000 else None,
        "min_samples_leaf": 2 if n_rows < 1_000 else 1,
    }


MODEL_REGISTRY: dict[str, ModelSpec] = {}


def _register(spec: ModelSpec) -> None:
    MODEL_REGISTRY[spec.key] = spec


_C = TaskType.CLASSIFICATION
_R = TaskType.REGRESSION

_register(ModelSpec(
    key="logistic_regression", name="Logistic Regression",
    estimators={_C: LogisticRegression},
    requires_scaling=True,
    hyperparameters=(
        HyperParam("max_iter", "int", 1000, min=50, max=100000, advanced=True),
    ),
    regularization={_C: RegularizationSpec(
        kind="penalty", options=("none", "l1", "l2", "elasticnet"),
        note="Strength is controlled by C (smaller C = stronger regularization).")},
    cost="low", interpretability="high",
    recommend=lambda n, f: {"max_iter": 1000},
))
_register(ModelSpec(
    key="knn", name="K-Nearest Neighbors",
    estimators={_C: KNeighborsClassifier},
    requires_scaling=True,
    hyperparameters=(
        HyperParam("n_neighbors", "int", 5, min=1, max=200),
        HyperParam("weights", "choice", "uniform", choices=("uniform", "distance")),
        HyperParam("metric", "choice", "minkowski",
                   choices=("minkowski", "euclidean", "manhattan"), advanced=True),
    ),
    regularization={_C: RegularizationSpec(
        kind="complexity", complexity_params=("n_neighbors",),
        note="Larger n_neighbors gives a smoother, less overfit decision boundary.")},
    cost="medium", interpretability="medium",
    recommend=lambda n, f: {"n_neighbors": max(3, min(25, int(n ** 0.5) | 1))},
))
_register(ModelSpec(
    key="decision_tree", name="Decision Tree",
    estimators={_C: DecisionTreeClassifier, _R: DecisionTreeRegressor},
    requires_scaling=False,
    hyperparameters=_tree_hparams(),
    regularization={_C: _TREE_REG, _R: _TREE_REG},
    cost="low", interpretability="high",
    recommend=lambda n, f: {"max_depth": 5 if n < 5_000 else 8, "min_samples_leaf": 5},
))
_register(ModelSpec(
    key="random_forest", name="Random Forest",
    estimators={_C: RandomForestClassifier, _R: RandomForestRegressor},
    requires_scaling=False,
    hyperparameters=_tree_hparams(_FOREST_EXTRA),
    regularization={_C: _TREE_REG, _R: _TREE_REG},
    cost="medium", interpretability="medium",
    recommend=_recommend_forest,
))
_register(ModelSpec(
    key="gradient_boosting", name="Gradient Boosting",
    estimators={_R: GradientBoostingRegressor},
    requires_scaling=False,
    hyperparameters=(
        HyperParam("n_estimators", "int", 100, min=10, max=1000),
        HyperParam("learning_rate", "float", 0.1, min=0.001, max=1.0),
        HyperParam("max_depth", "int", 3, min=1, max=20),
        HyperParam("subsample", "float", 1.0, min=0.1, max=1.0, advanced=True),
        HyperParam("min_samples_leaf", "int", 1, min=1, max=1000, advanced=True),
    ),
    regularization={_R: RegularizationSpec(
        kind="complexity",
        complexity_params=("learning_rate", "max_depth", "subsample", "min_samples_leaf"),
        note="Shrinkage (learning_rate), shallow trees and subsampling limit overfitting.")},
    cost="medium", interpretability="low",
    recommend=lambda n, f: {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
))
_register(ModelSpec(
    key="svm", name="Support Vector Machine",
    estimators={_C: SVC},
    requires_scaling=True,
    hyperparameters=(
        HyperParam("kernel", "choice", "rbf", choices=("linear", "rbf", "poly", "sigmoid")),
        HyperParam("gamma", "choice", "scale", choices=("scale", "auto"), advanced=True),
    ),
    regularization={_C: RegularizationSpec(
        kind="penalty", options=("l2",),
        note="Strength is controlled by C (smaller C = stronger regularization).")},
    cost="high", interpretability="low",
    recommend=lambda n, f: {"kernel": "rbf"},
))
_register(ModelSpec(
    key="linear_regression", name="Linear Regression",
    estimators={_R: LinearRegression},
    requires_scaling=False,
    hyperparameters=(HyperParam("fit_intercept", "bool", True, advanced=True),),
    regularization={_R: RegularizationSpec(
        kind="none", note="Ordinary least squares has no regularization. Use Ridge or Lasso.")},
    cost="low", interpretability="high",
))
_register(ModelSpec(
    key="ridge", name="Ridge Regression",
    estimators={_R: Ridge},
    requires_scaling=True,
    hyperparameters=(HyperParam("alpha", "float", 1.0, min=0.0, max=1e4),),
    regularization={_R: RegularizationSpec(kind="penalty", options=("l2",),
                                           note="Strength is controlled by alpha.")},
    cost="low", interpretability="high",
))
_register(ModelSpec(
    key="lasso", name="Lasso Regression",
    estimators={_R: Lasso},
    requires_scaling=True,
    hyperparameters=(
        HyperParam("alpha", "float", 1.0, min=1e-6, max=1e4),
        HyperParam("max_iter", "int", 5000, min=100, max=100000, advanced=True),
    ),
    regularization={_R: RegularizationSpec(kind="penalty", options=("l1",),
                                           note="Strength is controlled by alpha.")},
    cost="low", interpretability="high",
))


# --- public API -------------------------------------------------------------


def models_for_task(task: TaskType) -> list[ModelSpec]:
    return [m for m in MODEL_REGISTRY.values() if task in m.estimators]


def get_spec(key: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[key]
    except KeyError:
        raise ModelConfigError(f"Unknown model '{key}'. Known: {sorted(MODEL_REGISTRY)}") from None


def _validate_value(hp: HyperParam, value: Any, model: str) -> None:
    def bad(why: str) -> ModelConfigError:
        return ModelConfigError(f"{model}.{hp.name}={value!r}: {why}")

    if hp.kind == "choice":
        if value not in hp.choices:
            raise bad(f"must be one of {list(hp.choices)}")
    elif hp.kind == "bool":
        if not isinstance(value, bool):
            raise bad("must be true/false")
    elif hp.kind in ("int", "optional_int", "float"):
        if value is None and hp.kind == "optional_int":
            return
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise bad("must be a number")
        if hp.kind != "float" and int(value) != value:
            raise bad("must be an integer")
        if (hp.min is not None and value < hp.min) or (hp.max is not None and value > hp.max):
            raise bad(f"must be within [{hp.min}, {hp.max}]")


def validate_model_config(cfg: ModelConfig, task: TaskType) -> None:
    spec = get_spec(cfg.model_key)
    if task not in spec.estimators:
        raise ModelConfigError(f"{spec.name} does not support {task.value}.")
    allowed = {hp.name: hp for hp in spec.hyperparameters}
    reg_spec = spec.regularization[task]
    for name, value in cfg.hyperparameters.items():
        if name in allowed:
            _validate_value(allowed[name], value, spec.key)
        elif name not in reg_spec.complexity_params:
            raise ModelConfigError(f"{spec.name} has no hyperparameter '{name}'.")
    if cfg.regularization:
        if reg_spec.kind == "none":
            raise ModelConfigError(f"{spec.name} does not support regularization.")
        if reg_spec.kind == "complexity":
            raise ModelConfigError(
                f"{spec.name} is regularized via {list(reg_spec.complexity_params)}, "
                "not an L1/L2 penalty. Set those as hyperparameters.")
        rtype = cfg.regularization.get("type", reg_spec.options[0])
        if rtype not in reg_spec.options:
            raise ModelConfigError(
                f"{spec.name} supports regularization {list(reg_spec.options)}, got '{rtype}'.")


def recommended_hyperparameters(model_key: str, n_rows: int, n_features: int) -> dict[str, Any]:
    return dict(get_spec(model_key).recommend(n_rows, n_features))


def build_estimator(cfg: ModelConfig, task: TaskType, n_rows: int, n_features: int,
                    random_state: int):
    """Instantiate the sklearn estimator for a validated ModelConfig."""
    validate_model_config(cfg, task)
    spec = get_spec(cfg.model_key)
    params: dict[str, Any] = {hp.name: hp.default for hp in spec.hyperparameters
                              if hp.default is not None}
    if cfg.use_recommended_defaults:
        params.update(recommended_hyperparameters(cfg.model_key, n_rows, n_features))
    params.update(cfg.hyperparameters)
    if cfg.regularization:
        params.update(_PENALTY_TRANSLATORS[cfg.model_key](cfg.regularization))
    cls = spec.estimators[task]
    if "random_state" in cls().get_params():
        params["random_state"] = random_state
    return cls(**params)
