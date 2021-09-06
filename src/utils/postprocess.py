from npsem.NIPS2018POMIS_exp.test_bandit_strategies import compute_cumulative_regret, compute_optimality
import numpy as np


def find_optimal_arm_probability(arm_played, mu_star):
    return np.vectorize(lambda x: int(mu_star[x] == mu_star))(arm_played)


def get_results(arm_played, rewards, mu):
    results = dict()
    mu_star = np.max(mu)
    results["cumulative_regret"] = compute_cumulative_regret(rewards, mu_star)
    results["arm_optimality"] = compute_optimality(arm_played, mu)
    results["prob_arm_optimality"] = np.mean(results["arm_optimality"], axis=0)
    unique, counts = np.unique(arm_played, return_counts=True)
    results["frequency"] = dict(zip(unique, counts))

    return results
