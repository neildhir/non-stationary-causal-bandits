from npsem.model import StructuralCausalModel
from npsem.utils import combinations
from itertools import product
from typing import Dict, Tuple, Union, Any


def new_SCM_to_bandit_machine(
    M: StructuralCausalModel,
    interventions: list = None,
    reward_variable: str = "Y",
) -> tuple[Tuple, Dict[Union[int, Any], Dict]]:

    G = M.G
    mu_per_arm = list()  # Expected reward per arm
    arm_setting = dict()
    all_subsets = list(combinations(sorted(G.V - {reward_variable})))
    arm_id = 0

    if interventions:
        assert isinstance(interventions, list), interventions

    for subset in all_subsets:
        for values in product(*[M.D[variable] for variable in subset]):
            arm_setting[arm_id] = dict(zip(subset, values))
            if interventions:
                #  New way to intervene
                result = M.new_query(outcome=(reward_variable,), interventions=interventions + [arm_setting[arm_id]])
            else:
                #  Old way to intervene
                result = M.query(outcome=(reward_variable,), intervention=arm_setting[arm_id])
            expectation = sum(y_val * result[(y_val,)] for y_val in M.D[reward_variable])
            mu_per_arm.append(expectation)
            arm_id += 1

    return tuple(mu_per_arm), arm_setting
