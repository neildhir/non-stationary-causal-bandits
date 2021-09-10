from npsem.NIPS2018POMIS_exp.test_bandit_strategies import compute_cumulative_regret, compute_optimality
import numpy as np
from scipy.stats import bernoulli


def get_results(arm_played, rewards, mu):
    results = dict()
    mu_star = np.max(mu)
    results["cumulative_regret"] = compute_cumulative_regret(rewards, mu_star)
    results["arm_optimality"] = compute_optimality(arm_played, mu)
    results["prob_arm_optimality"] = np.mean(results["arm_optimality"], axis=0)
    unique, counts = np.unique(arm_played, return_counts=True)
    results["frequency"] = dict(zip(unique, counts))

    return results


def implement_intervention(causal_order, F, mu1, intervention, acausal=False):
    assert isinstance(intervention, dict)
    assert (
        len(intervention) == 1
    ), "The optimal intevention is multivariate: {}. Have not thought about that yet. Will deal with at a later stage.".format(
        intervention
    )

    for V_interv in intervention:
        break
    causal_order_idx = {var: causal_order.index(var) for var in causal_order}

    # This will eventually be our assigned time-slice.
    assigned = {key: bernoulli.rvs(success_prob) for (key, success_prob) in mu1.items()}
    for V_i in causal_order:
        #  We do not assign a-causal nodes
        if causal_order_idx[V_i] < causal_order_idx[V_interv]:
            assigned[V_i] = None
        else:
            if V_i in intervention:
                assigned[V_i] = intervention[V_i]
            elif acausal:
                # Assign a-causal nodes IF option invoked
                assigned[V_i] = F[V_i](assigned)
            else:
                assigned[V_i] = F[V_i](assigned)

    return assigned

