from __future__ import annotations

import math
from typing import Any

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_rollouts_per_condition(
    *,
    control_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = False,
    inflation: float = 1.15,
) -> dict[str, Any]:
    if not 0 <= control_rate < treatment_rate <= 1:
        raise ValueError("rates must satisfy 0 <= control_rate < treatment_rate <= 1")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be between zero and one")
    if inflation < 1:
        raise ValueError("inflation must be at least one")
    effect = abs(proportion_effectsize(treatment_rate, control_rate))
    alternative = "two-sided" if two_sided else "larger"
    raw = NormalIndPower().solve_power(
        effect_size=effect,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative=alternative,
    )
    adjusted = math.ceil(float(raw) * inflation)
    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_effect": treatment_rate - control_rate,
        "alpha": alpha,
        "power": power,
        "alternative": alternative,
        "unadjusted_rollouts_per_condition": math.ceil(float(raw)),
        "inflation_factor": inflation,
        "recommended_rollouts_per_condition": adjusted,
        "recommended_total_for_two_conditions": adjusted * 2,
        "note": (
            "This is a conservative independent-proportions approximation. Allocate the "
            "recommended runs evenly across goals, cases, topologies, and model families; "
            "replace assumed rates with blinded pilot estimates before freezing the design."
        ),
    }


def power_scenario_table() -> list[dict[str, Any]]:
    scenarios = [(0.05, 0.20), (0.05, 0.30), (0.10, 0.30), (0.10, 0.40), (0.20, 0.40)]
    return [
        required_rollouts_per_condition(control_rate=control, treatment_rate=treatment)
        for control, treatment in scenarios
    ]
