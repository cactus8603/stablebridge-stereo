from __future__ import annotations
from typing import Any
import numpy as np
SCORE_SELECTION="q10"
SCORE_NAME="benefit_minus_harm"

def policy_metrics(
    teacher: dict[str, np.ndarray], action: np.ndarray
) -> dict[str, Any]:
    gain = teacher["gain_abs_px"].astype(np.float64)
    realized = gain[np.arange(len(gain)), action]
    baseline = teacher["action_abs_px"][:, 0].astype(np.float64)
    oracle = np.maximum(teacher["oracle_gain_abs_px"].astype(np.float64), 0.0)
    condition = teacher["condition"].astype(str)
    family = teacher["family"].astype(str)
    scene = teacher["scene"].astype(str)
    clean = condition == "clean"
    corrupt = ~clean
    invoked = action != 0

    def improvement(mask: np.ndarray) -> float:
        return 100.0 * float(realized[mask].sum()) / float(baseline[mask].sum())

    condition_improvement = {
        name: improvement(corrupt & (condition == name))
        for name in sorted(set(condition[corrupt].tolist()))
    }
    family_improvement = {
        name: improvement(corrupt & (family == name))
        for name in sorted(set(family[corrupt].tolist()))
    }
    scene_improvement = {
        name: improvement(corrupt & (scene == name))
        for name in sorted(set(scene[corrupt].tolist()))
    }
    return {
        "corrupt_improvement_percent": improvement(corrupt),
        "corrupt_condition_macro_improvement_percent": float(
            np.mean(list(condition_improvement.values()))
        ),
        "oracle_gap_capture": float(realized[corrupt].sum() / oracle[corrupt].sum()),
        "corrupt_invocation_rate": float(invoked[corrupt].mean()),
        "corrupt_harmful_intervention_rate": float(
            np.mean(invoked[corrupt] & (realized[corrupt] < -0.001))
        ),
        "clean_improvement_percent": improvement(clean),
        "clean_invocation_rate": float(invoked[clean].mean()),
        "clean_harmful_intervention_rate": float(
            np.mean(invoked[clean] & (realized[clean] < -0.001))
        ),
        "positive_corrupt_conditions": int(
            sum(value > 0.0 for value in condition_improvement.values())
        ),
        "minimum_family_improvement_percent": float(min(family_improvement.values())),
        "worst_scene_improvement_percent": float(min(scene_improvement.values())),
        "condition_improvement_percent": condition_improvement,
        "family_improvement_percent": family_improvement,
        "action_counts": {
            str(index): int(np.sum(action[corrupt] == index)) for index in range(5)
        },
    }
def is_safe(metric: dict[str, Any]) -> bool:
    return bool(
        metric["clean_improvement_percent"] >= -0.2
        and metric["minimum_family_improvement_percent"] >= 0.0
        and metric["worst_scene_improvement_percent"] >= -0.5
    )


def compact(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "oracle_gap_capture": metric["oracle_gap_capture"],
        "corrupt_improvement_percent": metric["corrupt_improvement_percent"],
        "corrupt_condition_macro_improvement_percent": metric[
            "corrupt_condition_macro_improvement_percent"
        ],
        "corrupt_invocation_rate": metric["corrupt_invocation_rate"],
        "corrupt_harmful_intervention_rate": metric[
            "corrupt_harmful_intervention_rate"
        ],
        "clean_improvement_percent": metric["clean_improvement_percent"],
        "minimum_family_improvement_percent": metric[
            "minimum_family_improvement_percent"
        ],
        "worst_scene_improvement_percent": metric["worst_scene_improvement_percent"],
        "positive_corrupt_conditions": metric["positive_corrupt_conditions"],
        "family_improvement_percent": metric["family_improvement_percent"],
    }

def score_config(output):
    selected=output[:,:,1].argmax(axis=1)
    row=np.arange(len(output))
    benefit=1.0/(1.0+np.exp(-np.clip(output[row,selected,2],-30,30)))
    harm=1.0/(1.0+np.exp(-np.clip(output[row,selected,3],-30,30)))
    return {"selected":selected,"score":np.asarray(benefit-harm,dtype=np.float64)}
def calibrate_threshold(
    teacher: dict[str, np.ndarray], output: np.ndarray, coverage_points: int
) -> dict[str, Any]:
    config = score_config(output)
    order = np.argsort(-config["score"], kind="stable")
    counts = np.unique(np.rint(np.linspace(0, len(output), coverage_points)).astype(int))
    rows: list[dict[str, Any]] = []
    for count in counts:
        action = np.zeros(len(output), dtype=np.int64)
        chosen = order[:count]
        action[chosen] = config["selected"][chosen] + 1
        metric = policy_metrics(teacher, action)
        rows.append(
            {
                "invoked_rows": int(count),
                "score_threshold": float(config["score"][order[count - 1]])
                if count else float("inf"),
                "safe": is_safe(metric),
                "metrics": compact(metric),
            }
        )
    feasible = [row for row in rows if row["safe"]]
    best = max(
        feasible,
        key=lambda row: (
            row["metrics"]["oracle_gap_capture"],
            row["metrics"]["corrupt_condition_macro_improvement_percent"],
            -row["metrics"]["corrupt_harmful_intervention_rate"],
        ),
        default=None,
    )
    return {
        "selection": SCORE_SELECTION,
        "score_name": SCORE_NAME,
        "grid_points": len(rows),
        "feasible_points": len(feasible),
        "best": best,
    }


def apply_threshold(output: np.ndarray, threshold: float) -> np.ndarray:
    config = score_config(output)
    allowed = config["score"] >= threshold
    return np.where(allowed, config["selected"] + 1, 0).astype(np.int64)


def scene_bootstrap(
    teacher: dict[str, np.ndarray], action: np.ndarray, replicates: int
) -> dict[str, Any]:
    gain = teacher["gain_abs_px"].astype(np.float64)
    realized = gain[np.arange(len(gain)), action]
    baseline = teacher["action_abs_px"][:, 0].astype(np.float64)
    oracle = np.maximum(teacher["oracle_gain_abs_px"].astype(np.float64), 0.0)
    condition = teacher["condition"].astype(str)
    scene = teacher["scene"].astype(str)
    corrupt_conditions = sorted(set(condition[condition != "clean"].tolist()))
    scenes = sorted(set(scene.tolist()))
    scene_lookup = {name: index for index, name in enumerate(scenes)}
    scene_index = np.asarray([scene_lookup[name] for name in scene], dtype=np.int64)
    realized_condition = np.zeros((len(scenes), len(corrupt_conditions)), dtype=np.float64)
    baseline_condition = np.zeros_like(realized_condition)
    realized_corrupt = np.zeros(len(scenes), dtype=np.float64)
    oracle_corrupt = np.zeros(len(scenes), dtype=np.float64)
    corrupt = condition != "clean"
    for scene_id in range(len(scenes)):
        scene_mask = scene_index == scene_id
        realized_corrupt[scene_id] = realized[scene_mask & corrupt].sum()
        oracle_corrupt[scene_id] = oracle[scene_mask & corrupt].sum()
        for condition_id, name in enumerate(corrupt_conditions):
            mask = scene_mask & (condition == name)
            realized_condition[scene_id, condition_id] = realized[mask].sum()
            baseline_condition[scene_id, condition_id] = baseline[mask].sum()
    rng = np.random.default_rng(20260824)
    capture = np.empty(replicates, dtype=np.float64)
    macro = np.empty(replicates, dtype=np.float64)
    condition_values = np.empty((replicates, len(corrupt_conditions)), dtype=np.float64)
    for index in range(replicates):
        weights = np.bincount(
            rng.integers(0, len(scenes), size=len(scenes)), minlength=len(scenes)
        ).astype(np.float64)
        capture[index] = (weights @ realized_corrupt) / max(weights @ oracle_corrupt, 1e-12)
        numerator = weights @ realized_condition
        denominator = np.maximum(weights @ baseline_condition, 1e-12)
        condition_values[index] = 100.0 * numerator / denominator
        macro[index] = condition_values[index].mean()
    interval = lambda value: [
        float(np.quantile(value, 0.025)), float(np.quantile(value, 0.975))
    ]
    condition_ci = {
        name: interval(condition_values[:, number])
        for number, name in enumerate(corrupt_conditions)
    }
    return {
        "unit": "scene",
        "replicates": replicates,
        "oracle_gap_capture_95ci": interval(capture),
        "corrupt_condition_macro_improvement_percent_95ci": interval(macro),
        "condition_improvement_percent_95ci": condition_ci,
        "conditions_with_significant_regression": sorted(
            name for name, bounds in condition_ci.items() if bounds[1] < 0.0
        ),
    }
