from npsem.NIPS2018POMIS_exp.test_bandit_strategies import compute_cumulative_regret, compute_optimality
from npsem.model import StructuralCausalModel
import numpy as np


def get_results(arm_played, rewards, mu):
    results = dict()
    mu_star = np.max(mu)
    results["cumulative_regret"] = compute_cumulative_regret(rewards, mu_star)
    results["arm_optimality"] = compute_optimality(arm_played, mu)
    results["prob_arm_optimality"] = np.mean(results["arm_optimality"], axis=0)
    unique, counts = np.unique(arm_played, return_counts=True)
    results["frequency"] = dict(zip(unique, counts))

    return results


def assign_blanket(M: StructuralCausalModel, blanket: dict, best_intervention: dict, target_var_only: str,) -> dict:

    assert isinstance(best_intervention, dict)

    # Assign target value
    causal_effect = M.query(outcome=(target_var_only,), intervention=best_intervention)
    # Greedily pick the outcome value with the highest probability under the SCM model
    blanket[target_var_only] = max(causal_effect, key=causal_effect.get)[0]

    if best_intervention:
        # Assign best intervention
        for (key, value) in best_intervention.items():
            blanket[key] = value

    # XXX: note that nodes which are not in {outcome_var, interventions_vars} are _not_ assigned a value (currently). It is at present not clear what we should do with these variables. We could perhaps draw a sample from them and then use that.

    return blanket
