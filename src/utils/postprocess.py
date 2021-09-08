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
    # TODO: this is almost surely not correct, it should be the expected value OR more likely just the value of the SEM executed for the samples values of U plus the value of intervention. We probably also need to use the same samples of U throughout the whole process.
    blanket[target_var_only] = max(causal_effect, key=causal_effect.get)[0]

    if best_intervention:
        # Assign best intervention
        for (key, value) in best_intervention.items():
            blanket[key] = value

        #  Assign un-assigned variables IF they are of higher causal order than the intervention
        unassigned_vars = [key for key in blanket.keys() if blanket[key] == None]
        if len(unassigned_vars) > 0:
            vars_to_assign = []
            if len(best_intervention) == 1:
                interv_idx = M.G.causal_order().index(key)
            else:
                raise NotImplementedError("The optimal intevention is multivariate. Have not thought about that yet.")
            #  Check if the un-assigned nodes are of higher causal order
            for var in unassigned_vars:
                assert var != target_var_only
                var_idx = M.G.causal_order().index(var)
                #  If the order is higher than the intervention then we can assign it a value
                if interv_idx < var_idx:
                    vars_to_assign.append(var)

        if len(vars_to_assign) > 0:
            # This whole function needs to be a recursion since we need to update the list each time we assign a node with higher causal order.
            pass
        else:
            return blanket

        # XXX: note that nodes which are not in {outcome_var, interventions_vars} are _not_ assigned a value (currently). It is at present not clear what we should do with these variables. We could perhaps draw a sample from them and then use that.
    else:
        # This is option exist for rare situations where the empty set (i.e. best_intervention = {}) is the best intervention.
        return blanket
