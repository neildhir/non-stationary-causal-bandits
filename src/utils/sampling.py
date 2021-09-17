import numpy as np
from scipy.stats import bernoulli


def sample_sem(
    sem, exogenous_and_confounder_probs: dict, timesteps: int, sample_size: int, seed=0,
):
    """
    Draw sample(s) from given structural equation model under steady-state condition (i.e. we do not fix any nodes as we walk along the graph).

    Returns
    -------
    dict
        Sequential sample(s) from SEM.
    """

    SEM = sem()
    #  We use the vectorised version where we can sample N >= 1 per call, as this is faster.
    static = SEM.static_vec()
    dynamic = SEM.dynamic_vec()

    assert static.keys() == dynamic.keys()

    if seed:
        np.random.seed(seed)

    # Pre-allocate the sample container
    empty_samples = {k: np.empty((sample_size, timesteps), dtype="int") for k in static.keys()}
    background_samples = sample_binary_exogenous_and_confounders(
        exogenous_and_confounder_probs, N=sample_size, T=timesteps
    )
    # Combine dictionaries
    samples = empty_samples | background_samples

    for t in range(timesteps):
        # Allocate SEM according to time index
        model = static if t == 0 else dynamic
        for endogenous_var, function in model.items():
            samples[endogenous_var][:, t] = function(samples, t)

    return samples


def sample_binary_exogenous_and_confounders(probs: dict, N: int, T: int) -> dict:
    """
    Uses a Bernoulli distribution as a generator.

    Parameters
    ----------
    probs : dict
        Dictionary of probablities for background (exogenous) and confounders
    N : int
        Sample count per variable, per time-step
    T : int
        Number of time-steps in graph

    Returns
    -------
    dict
        Dictionary containing samples for exogenous variables.
    """
    return {U_i: bernoulli.rvs(p=prob, size=(N, T)) for (U_i, prob) in probs.items()}

