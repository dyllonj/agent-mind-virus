from mindvirus.power import required_rollouts_per_condition


def test_power_increases_for_smaller_effect() -> None:
    smaller = required_rollouts_per_condition(control_rate=0.05, treatment_rate=0.20)
    larger = required_rollouts_per_condition(control_rate=0.05, treatment_rate=0.30)
    assert (
        smaller["recommended_rollouts_per_condition"] > larger["recommended_rollouts_per_condition"]
    )
