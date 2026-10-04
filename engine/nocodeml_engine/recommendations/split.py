"""Deterministic split recommendations (Section 3). Rules over the dataset profile only."""

from __future__ import annotations

from nocodeml_engine.config import PipelineConfig, SplitConfig, SplitMethod, TaskType
from nocodeml_engine.profiling.profiler import DatasetProfile
from nocodeml_engine.recommendations.preprocessing import Recommendation

SMALL_DATASET = 1000


def recommend_split(profile: DatasetProfile, task: TaskType, n_rows: int) -> list[Recommendation]:
    """Each action is {"type": "split", "patch": {...SplitConfig fields...}}; patches merge in order."""
    recs: list[Recommendation] = []
    imbalanced = bool(
        task is TaskType.CLASSIFICATION and profile.target
        and profile.target.get("class_distribution", {}).get("imbalanced"))

    # 1. How to split
    if n_rows < SMALL_DATASET:
        method = SplitMethod.STRATIFIED_K_FOLD if task is TaskType.CLASSIFICATION else SplitMethod.K_FOLD
        label = "Stratified 5-fold cross-validation" if imbalanced or task is TaskType.CLASSIFICATION \
            else "5-fold cross-validation"
        recs.append(Recommendation(
            "split_method", "split_method", None, label,
            f"With only {n_rows} rows, a single train/test split is a noisy estimate: a few unlucky rows can "
            "swing the score. Cross-validation trains and tests five times so every row is used for testing once.",
            0.8, {"type": "split", "patch": {"method": method.value, "n_splits": 5, "stratify": False}}))
    else:
        recs.append(Recommendation(
            "split_method", "split_method", None, "Train / test split, 80% / 20%",
            f"With {n_rows:,} rows, holding out 20% leaves plenty to test on and plenty to train on, "
            "and it is the fastest option.", 0.85,
            {"type": "split", "patch": {"method": "train_test", "test_size": 0.2, "stratify": False}}))
        method = SplitMethod.TRAIN_TEST

    # 2. Keep class proportions when classes are imbalanced (already implied by stratified k-fold)
    if imbalanced and method is not SplitMethod.STRATIFIED_K_FOLD:
        ratio = profile.target["class_distribution"]["minority_ratio"]
        recs.append(Recommendation(
            "stratify", "stratify", None, "Stratify the split",
            f"Your smallest class is only {100 * ratio:.1f}% of rows. Stratifying keeps that proportion in both the "
            "training and test sets, so the test set is not left with too few examples of it.", 0.9,
            {"type": "split", "patch": {"stratify": True}}))

    # 3. Time-ordered data (a judgement call, so optional)
    if profile.datetime:
        col = profile.datetime[0]
        recs.append(Recommendation(
            "time_split", "time_split", col, f"Split chronologically by '{col}'",
            f"Your data has a date column ('{col}'). If you are predicting the future from the past, a random split "
            "lets the model peek at later rows while training and the score will look better than it really is. "
            "A chronological split trains on earlier rows and tests on later ones. Skip this if row order doesn't "
            "matter for your problem.", 0.5,
            {"type": "split", "patch": {"method": "train_test", "time_column": col, "stratify": False}},
            optional=True))
    return recs


def apply_split_actions(config: PipelineConfig, actions: list[dict]) -> PipelineConfig:
    """Reference implementation of what the web UI does with a ticked set of recommendations."""
    patch: dict = {}
    for a in actions:
        if a.get("type") != "split":
            raise ValueError(f"Unknown recommendation action '{a.get('type')}'")
        patch.update(a["patch"])
    return config.model_copy(update={"split": SplitConfig(**{**config.split.model_dump(), **patch})})
