from npsem.NIPS2018POMIS_exp.test_bandit_strategies import compute_cumulative_regret, compute_optimality
from npsem.model import StructuralCausalModel
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


def assign_blanket(M: StructuralCausalModel, blanket: dict, best_intervention: dict, target_var_only: str,) -> dict:

    assert isinstance(best_intervention, dict)

    # Assign target value
    causal_effect = M.query(outcome=(target_var_only,), intervention=best_intervention)
    # Greedily pick the outcome value with the highest probability under the SCM model
    # TODO: this is almost surely not correct, it should be the expected value OR more likely just the value of the SEM executed for the samples values of U plus the value of intervention. We probably also need to use the same samples of U throughout the whole process.
    # TODO: this has the highest causal order so we should move it to the end of the recursion below.
    blanket[target_var_only] = max(causal_effect, key=causal_effect.get)[0]

    if best_intervention:
        # Assign best intervention
        for (key, value) in best_intervention.items():
            blanket[key] = value

        #  Assign un-assigned variables IF they are of higher causal order than the intervention
        unassigned_vars = [key for key in blanket.keys() if blanket[key] == None]
        #  We could compute this in the main function but we don't since the order could change per time-slice
        causal_order = {var: M.G.causal_order().index(var) for var in blanket.keys() if var != target_var_only}
        if len(unassigned_vars) > 0:
            if len(best_intervention) == 1:
                for key in best_intervention.keys():
                    interv_idx = causal_order[key]
            else:
                raise NotImplementedError("The optimal intevention is multivariate. Have not thought about that yet.")

            #  Check if the un-assigned nodes are of higher causal order
            for var in unassigned_vars:
                if causal_order[var] > interv_idx:
                    pass
        else:
            return blanket

        # XXX: note that nodes which are not in {outcome_var, interventions_vars} are _not_ assigned a value (currently). It is at present not clear what we should do with these variables. We could perhaps draw a sample from them and then use that.
    else:
        # This is option exist for rare situations where the empty set (i.e. best_intervention = {}) is the best intervention.
        return blanket


def implement_intervention(causal_order, F, mu1, best_intervention):
    assert isinstance(best_intervention, dict)
    assert (
        len(best_intervention) == 1
    ), "The optimal intevention is multivariate. Have not thought about that yet. Will deal with at a later stage."

    for V_interv in best_intervention:
        break
    causal_order_idx = {var: causal_order.index(var) for var in causal_order}

    # This will eventually be our assigned time-slice.
    assigned = {key: bernoulli.rvs(success_prob) for (key, success_prob) in mu1.items()}
    for V_i in causal_order:
        #  We do not assign acausal nodes
        if causal_order_idx[V_i] < causal_order_idx[V_interv]:
            assigned[V_i] = None
        else:
            if V_i in best_intervention:
                assigned[V_i] = best_intervention[V_i]
            else:
                assigned[V_i] = F[V_i](assigned)

    return assigned

