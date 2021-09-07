from typing import OrderedDict
from npsem.NIPS2018POMIS_exp.test_bandit_strategies import compute_cumulative_regret, compute_optimality
import numpy as np
from utils.sampling import sample_sem


def get_results(arm_played, rewards, mu):
    results = dict()
    mu_star = np.max(mu)
    results["cumulative_regret"] = compute_cumulative_regret(rewards, mu_star)
    results["arm_optimality"] = compute_optimality(arm_played, mu)
    results["prob_arm_optimality"] = np.mean(results["arm_optimality"], axis=0)
    unique, counts = np.unique(arm_played, return_counts=True)
    results["frequency"] = dict(zip(unique, counts))

    return results


def assign_blanket(
    blanket: dict(dict),
    temporal_index: int,
    best_intervention: dict,
    target_var_only: str,
    target_value: float,
    transition_funcs: dict = None,
):

    """
    This whole routine needs to be a combination between SCM_to_bandit_machine and query00 of SCM.
    """

    assert isinstance(best_intervention, dict)

    if best_intervention:
        # Assign best intervention
        for (key, value) in best_intervention.items():
            blanket[temporal_index][key] = value

    # Assign target value
    blanket[temporal_index][target_var_only] = target_value

    return blanket
