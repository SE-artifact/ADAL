from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .config import AggregationConfig, METHOD_VERSION


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    monotonic: int


FEATURE_SPECS = [
    FeatureSpec("prior_score", 1),
    FeatureSpec("raw_rank_feature", 0),
    FeatureSpec("support_score", 1),
    FeatureSpec("base_support_score", 1),
    FeatureSpec("semantic_support_score", 1),
    FeatureSpec("strong_schema_count", 1),
    FeatureSpec("resolved_rebuttal_count", 1),
    FeatureSpec("resolved_high_medium_count", 1),
    FeatureSpec("identifier_overlap", 1),
    FeatureSpec("operands_preserved", 1),
    FeatureSpec("null_guard_before_deref", 1),
    FeatureSpec("condition_flip_explains_symptom", 1),
    FeatureSpec("return_contract_alignment", 1),
    FeatureSpec("def_use_role_support", 1),
    FeatureSpec("controlled_region_guard", 1),
    FeatureSpec("trigger_call_alignment", 1),
    FeatureSpec("supported_high_count", -1),
    FeatureSpec("supported_medium_count", -1),
    FeatureSpec("unresolved_high_medium_count", -1),
    FeatureSpec("rebuttal_penalty", -1),
    FeatureSpec("missing_penalty", -1),
    FeatureSpec("missing_count", -1),
    FeatureSpec("public_regression_missing", -1),
    FeatureSpec("side_effect_deleted", -1),
    FeatureSpec("condition_contradiction", -1),
    FeatureSpec("return_contract_risk", -1),
    FeatureSpec("receiver_delta", -1),
    FeatureSpec("variable_to_constant", -1),
    FeatureSpec("expression_operand_loss", -1),
    FeatureSpec("trigger_literal_overlap", -1),
    FeatureSpec("patch_size", 0),
    FeatureSpec("touches_buggy_method", 1),
    FeatureSpec("touches_buggy_line", 1),
    FeatureSpec("method_public_or_protected", -1),
    FeatureSpec("type_condition_change", 0),
    FeatureSpec("type_guard_insertion", 0),
    FeatureSpec("type_null_check_change", 0),
    FeatureSpec("type_constant_change", 0),
    FeatureSpec("type_api_call_change", 0),
    FeatureSpec("type_deletion", 0),
    FeatureSpec("type_value_change", 0),
]

FEATURE_NAMES = [item.name for item in FEATURE_SPECS]
MONOTONIC_CONSTRAINTS = [item.monotonic for item in FEATURE_SPECS]
PATCH_TYPE_FEATURES = {
    "type_condition_change", "type_guard_insertion", "type_null_check_change",
    "type_constant_change", "type_api_call_change", "type_deletion", "type_value_change",
}
RANK_FEATURE_NAMES = [name for name in FEATURE_NAMES if name not in PATCH_TYPE_FEATURES]
ACTION_SIGNAL = {
    ("ranking_boost", "low"): 0.20,
    ("ranking_boost", "medium"): 0.20,
    ("ranking_boost", "high"): 0.35,
    ("promote_to_candidate", "low"): 1.60,
    ("promote_to_candidate", "medium"): 1.60,
    ("promote_to_candidate", "high"): 2.20,
}


def semantic_signal(row: dict[str, Any]) -> float:
    revision = row.get("semantic_revision", {})
    raw = ACTION_SIGNAL.get(
        (
            revision.get("effective_action", "no_change"),
            revision.get("effective_action_confidence", "low"),
        ),
        0.0,
    )
    return min(1.0, max(0.0, float(raw) / 2.20))


def _surviving_rebuttals(row: dict[str, Any], severity: str) -> int:
    return sum(
        item.get("severity") == severity
        and item.get("status") in {"supported", "unresolved"}
        for item in row.get("evidence_graph", {}).get("rebuttals", [])
    )


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    for row in rows:
        missing = set(names) - set(row.get("feature_vector", {}))
        if missing:
            raise ValueError("Missing evidence features: " + ", ".join(sorted(missing)))
    return np.asarray(
        [[float(row.get("feature_vector", {}).get(name, 0.0)) for name in names] for row in rows],
        dtype=float,
    )


def _labels(rows: list[dict[str, Any]]) -> np.ndarray:
    if any(row.get("truth_label") not in {"correct", "incorrect"} for row in rows):
        raise ValueError("Every training row must have a correct/incorrect truth_label.")
    return np.asarray([int(row.get("truth_label") == "correct") for row in rows], dtype=int)


def _bugs(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([str(row.get("project", "")) + "::" + str(row.get("bug_uid", "")) for row in rows])


def _pool_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("project", "")), str(row.get("tool", "")), str(row.get("bug_uid", ""))


def _filter_model(seed: int, config: AggregationConfig | None = None) -> HistGradientBoostingClassifier:
    config = config or AggregationConfig()
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        min_samples_leaf=config.min_samples_leaf,
        l2_regularization=config.filter_l2,
        monotonic_cst=MONOTONIC_CONSTRAINTS,
        random_state=seed,
    )


def _group_folds(groups: np.ndarray, seed: int, n_splits: int = 3):
    unique, counts = np.unique(groups, return_counts=True)
    random = np.random.RandomState(seed)
    order = random.permutation(len(unique))
    ordered = sorted(order, key=lambda index: -counts[index])
    folds: list[list[str]] = [[] for _ in range(n_splits)]
    sizes = [0] * n_splits
    for index in ordered:
        fold = min(range(n_splits), key=lambda value: (sizes[value], value))
        folds[fold].append(unique[index])
        sizes[fold] += int(counts[index])
    indices = np.arange(len(groups))
    for validation_groups in folds:
        validation = np.isin(groups, validation_groups)
        yield indices[~validation], indices[validation]


def _pairwise_matrix(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    max_pairs_per_bug: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    labels = _labels(rows)
    features = _matrix(rows, RANK_FEATURE_NAMES)
    for index in indices:
        grouped[_pool_key(rows[int(index)])].append(int(index))
    x_values = []
    y_values = []
    for key in sorted(grouped):
        pool = grouped[key]
        correct = [index for index in pool if labels[index] == 1]
        incorrect = [index for index in pool if labels[index] == 0]
        count = 0
        for correct_index in correct:
            for incorrect_index in incorrect:
                delta = features[correct_index] - features[incorrect_index]
                x_values.extend([delta, -delta])
                y_values.extend([1, 0])
                count += 1
                if max_pairs_per_bug is not None and count >= max_pairs_per_bug:
                    break
            if max_pairs_per_bug is not None and count >= max_pairs_per_bug:
                break
    if not x_values:
        raise ValueError("Ranking requires at least one correct-incorrect pair.")
    return np.asarray(x_values), np.asarray(y_values)


def _ranking_model(rows: list[dict[str, Any]], indices: np.ndarray, seed: int, config: AggregationConfig | None = None) -> LogisticRegression:
    config = config or AggregationConfig()
    x_pair, y_pair = _pairwise_matrix(rows, indices, config.max_pairs_per_pool)
    model = LogisticRegression(
        penalty="l2",
        C=config.ranking_C,
        fit_intercept=False,
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=seed,
    )
    model.fit(x_pair, y_pair)
    return model


def _far(y: np.ndarray, probability: np.ndarray, threshold: float) -> float:
    prediction = probability >= threshold
    false_positive = float(np.sum(prediction & (y == 0)))
    negative = float(np.sum(y == 0))
    return false_positive / negative if negative else 0.0


def _filter_threshold(y: np.ndarray, probability: np.ndarray, target_far: float) -> float:
    if not 0 <= target_far <= 1:
        raise ValueError("The false-acceptance budget must be in [0,1].")
    candidates = sorted(set(probability.tolist()) | {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0})
    feasible = []
    for threshold in candidates:
        prediction = probability >= threshold
        recall = float(np.sum(prediction & (y == 1))) / max(1.0, float(np.sum(y == 1)))
        precision = float(np.sum(prediction & (y == 1))) / max(1.0, float(np.sum(prediction)))
        far = _far(y, probability, threshold)
        if far <= target_far:
            feasible.append((recall, precision, -far, -threshold))
    return -max(feasible)[3] if feasible else 0.5


def _pool_minmax(rows: list[dict[str, Any]], values: np.ndarray) -> np.ndarray:
    output = np.zeros(len(values), dtype=float)
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[_pool_key(row)].append(index)
    for indices in grouped.values():
        current = values[indices]
        lower = float(np.min(current))
        upper = float(np.max(current))
        if upper > lower:
            output[indices] = (current - lower) / (upper - lower)
    return output


def _ranking_objective(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    probability: np.ndarray,
    filter_threshold: float,
) -> tuple[int, int, int, int]:
    grouped: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if probability[index] >= filter_threshold:
            grouped[_pool_key(row)].append(index)
    top1 = 0
    top5 = 0
    effort = 0
    effort_all = 0
    for indices in grouped.values():
        ordered = sorted(
            indices,
            key=lambda index: (
                -scores[index],
                -probability[index],
                _surviving_rebuttals(rows[index], "high"),
                _surviving_rebuttals(rows[index], "medium"),
                int(rows[index].get("patch_rank", 0)),
                str(rows[index].get("patch_uid", "")),
            ),
        )
        positions = [rank for rank, index in enumerate(ordered, 1) if rows[index].get("truth_label") == "correct"]
        if positions:
            top1 += int(positions[0] == 1)
            top5 += int(positions[0] <= 5)
            effort += positions[0]
            effort_all += positions[-1]
        else:
            effort += len(ordered)
            effort_all += len(ordered)
    return top5, top1, -effort, -effort_all


class ADALAggregator:
    def __init__(self, seed: int = 43, config: AggregationConfig | None = None):
        self.seed = seed
        self.config = config or AggregationConfig()
        self.filter_head: HistGradientBoostingClassifier | None = None
        self.calibrator: LogisticRegression | None = None
        self.ranking_head: LogisticRegression | None = None
        self.filter_threshold = 0.5
        self.blend_alpha = 0.0
        self.training_record: dict[str, Any] = {}

    def fit(self, rows: list[dict[str, Any]], target_far: float | None = None) -> "ADALAggregator":
        y, groups = _labels(rows), _bugs(rows)
        if len(set(groups.tolist())) < 3 or len(set(y.tolist())) < 2:
            raise ValueError("Training requires at least three bugs and both labels.")
        target_far = self.config.target_false_acceptance_rate if target_far is None else target_far
        indices = np.arange(len(rows))
        oof_filter, oof_rank = np.zeros(len(rows)), np.zeros(len(rows))
        x_filter, x_rank = _matrix(rows, FEATURE_NAMES), _matrix(rows, RANK_FEATURE_NAMES)
        folds = []
        for fold, (train_index, validation_index) in enumerate(_group_folds(groups, self.seed), 1):
            filtering = _filter_model(self.seed, self.config)
            filtering.fit(x_filter[train_index], y[train_index])
            ranking = _ranking_model(rows, train_index, self.seed, self.config)
            oof_filter[validation_index] = filtering.predict_proba(x_filter[validation_index])[:, 1]
            oof_rank[validation_index] = x_rank[validation_index] @ ranking.coef_.ravel()
            folds.append({
                "fold": fold, "training_patch_ids": [rows[int(i)]["patch_uid"] for i in train_index],
                "validation_patch_ids": [rows[int(i)]["patch_uid"] for i in validation_index],
                "training_bugs": sorted(set(groups[train_index].tolist())),
                "validation_bugs": sorted(set(groups[validation_index].tolist())),
            })
        # Threefold OOF sigmoid calibration is fitted exclusively on outer-training
        # projects. It does not use an in-sample final-filter score or a test label.
        self.calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000, random_state=self.seed)
        self.calibrator.fit(oof_filter.reshape(-1, 1), y)
        constrained_to_constant = bool(self.calibrator.coef_[0, 0] < 0)
        if constrained_to_constant:
            # A negative sigmoid slope would reverse every monotonic evidence direction.
            prevalence = float(np.clip(np.mean(y), 1e-12, 1.0 - 1e-12))
            self.calibrator.coef_[0, 0] = 0.0
            self.calibrator.intercept_[0] = np.log(prevalence / (1.0 - prevalence))
        calibrated = self.calibrator.predict_proba(oof_filter.reshape(-1, 1))[:, 1]
        self.filter_threshold = _filter_threshold(y, calibrated, target_far)
        rank_normalized = _pool_minmax(rows, oof_rank)
        probability_normalized = _pool_minmax(rows, calibrated)
        semantic = np.asarray([semantic_signal(row) for row in rows], dtype=float)
        candidates = sorted(set(self.config.alpha_grid))
        if not candidates or any(alpha < 0 for alpha in candidates):
            raise ValueError("alpha_grid must contain non-negative coefficients.")
        self.blend_alpha = max(
            candidates,
            key=lambda alpha: (
                *_ranking_objective(rows, rank_normalized + alpha * probability_normalized + semantic,
                                    calibrated, self.filter_threshold),
                -alpha,
            ),
        )
        self.filter_head = _filter_model(self.seed, self.config)
        self.filter_head.fit(x_filter, y)
        self.ranking_head = _ranking_model(rows, indices, self.seed, self.config)
        self.training_record = {
            "method_version": METHOD_VERSION,
            "seed": self.seed, "parameters": asdict(self.config),
            "projects": sorted({row.get("project", "") for row in rows}),
            "training_patch_ids": [row["patch_uid"] for row in rows],
            "calibration": "sigmoid on threefold bug-grouped OOF filtering probabilities",
            "sigmoid_slope": float(self.calibrator.coef_[0, 0]),
            "sigmoid_intercept": float(self.calibrator.intercept_[0]),
            "nonnegative_slope_boundary_used": constrained_to_constant,
            "inner_folds": folds, "filter_threshold": self.filter_threshold, "alpha": self.blend_alpha,
            "alpha_objective": ["Top-5", "Top-1", "-RE@1", "-RE@all", "-alpha"],
            "filter_threshold_objective": "maximize recall, precision, -FAR, -threshold subject to training OOF FAR budget",
            "feature_names": FEATURE_NAMES, "ranking_features": RANK_FEATURE_NAMES,
            "monotonic_constraints": dict(zip(FEATURE_NAMES, MONOTONIC_CONSTRAINTS)),
        }
        return self

    def predict(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.filter_head is None or self.calibrator is None or self.ranking_head is None:
            raise ValueError("The aggregator is not fitted.")
        if not rows:
            return []
        raw_probability = self.filter_head.predict_proba(_matrix(rows, FEATURE_NAMES))[:, 1]
        probability = self.calibrator.predict_proba(raw_probability.reshape(-1, 1))[:, 1]
        raw_rank = _matrix(rows, RANK_FEATURE_NAMES) @ self.ranking_head.coef_.ravel()
        ranking = _pool_minmax(rows, raw_rank)
        ranking += self.blend_alpha * _pool_minmax(rows, probability)
        ranking += np.asarray([semantic_signal(row) for row in rows], dtype=float)
        output = []
        for row, prob, utility, score in zip(rows, probability, raw_rank, ranking):
            item = dict(row)
            item.update({
                "filter_probability": float(prob), "ranking_utility": float(utility),
                "ranking_score": float(score), "semantic_revision_signal": semantic_signal(row),
                "filter_threshold": self.filter_threshold, "fusion_alpha": self.blend_alpha,
                # Equation (9) ranks; the threshold on calibrated p_c alone classifies.
                "predicted_label": "correct" if prob >= self.filter_threshold else "incorrect",
                "assessment_status": "calibrated_dual_head",
            })
            output.append(item)
        return output

    def save(self, path: str) -> None:
        if not path:
            raise ValueError("Model output path is empty.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        Path(path).with_suffix(".training.json").write_text(
            json.dumps(self.training_record, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> "ADALAggregator":
        if not path:
            raise ValueError("Model path is empty.")
        value = joblib.load(path)
        if not isinstance(value, cls):
            raise TypeError("The model file does not contain an ADAL aggregator.")
        if getattr(value, "training_record", {}).get("method_version") != METHOD_VERSION:
            raise ValueError("The checkpoint belongs to a different method version; retrain it with the aligned implementation.")
        return value


def lopo_predictions(
    rows: list[dict[str, Any]], seed: int = 43, config: AggregationConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, ADALAggregator]]:
    projects = sorted({str(row.get("project", "")) for row in rows})
    if len(projects) < 2 or "" in projects:
        raise ValueError("LOPO requires at least two non-empty project identifiers.")
    if len({row["patch_uid"] for row in rows}) != len(rows):
        raise ValueError("LOPO requires unique patch identifiers.")
    predictions, models = [], {}
    for project in projects:
        training = [row for row in rows if row["project"] != project]
        testing = [row for row in rows if row["project"] == project]
        model = ADALAggregator(seed, config).fit(training)
        model.training_record["heldout_project"] = project
        model.training_record["test_patch_ids"] = [row["patch_uid"] for row in testing]
        for prediction in model.predict(testing):
            prediction["heldout_project"] = project
            predictions.append(prediction)
        models[project] = model
    return predictions, models
