from numpy import vectorize
from npsem.bandits import play_bandits
from npsem.model import StructuralCausalModel
from npsem.scm_bandits import arms_of, new_SCM_to_bandit_machine
from npsem.utils import subseq


def main_experiment_ccb(M: StructuralCausalModel, Y, past_interventions=None, num_trial=200, horizon=10000, n_jobs=1):
    results = dict()
    mu, arm_setting = new_SCM_to_bandit_machine(scm, interventions=past_interventions, reward_variable=target_var_only)
    arm_strategy = "POMIS"
    arm_selected = arms_of(arm_strategy, arm_setting, M.G, Y)
    arm_corrector = vectorize(lambda x: arm_selected[x])
    for bandit_algo in ["TS", "UCB"]:
        arm_played, rewards = play_bandits(horizon, subseq(mu, arm_selected), bandit_algo, num_trial, n_jobs)
        results[(arm_strategy, bandit_algo)] = arm_corrector(arm_played), rewards

    return results, mu
