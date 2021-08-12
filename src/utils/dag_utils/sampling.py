from collections import OrderedDict
import numpy as np


def sample_sem(
    sem, exo_vars: dict, timesteps: int, sample_size: int, epsilon=None, seed=None,
):
    """
    Draw sample(s) from given structural equation model.

    Returns
    -------
    dict
        Sequential sample(s) from SEM.
    """

    SEM = sem()
    static = SEM.static()
    dynamic = SEM.dynamic()

    assert static.keys() == dynamic.keys()

    if seed:
        np.random.seed(seed)

    if epsilon:
        assert epsilon.shape == (sample_size, timesteps)
        assert all(exo_vars[u].shape == exo_vars[0].shape for u in exo_vars.keys())
        assert exo_vars[0].shape == epsilon.shape

    # Pre-allocate the sample container
    sem_samples = {k: np.empty((sample_size, timesteps), dtype="int") for k in static.keys()}

    for t in range(timesteps):
        model = static if t == 0 else dynamic
        for var, function in model.items():
            sem_samples[var][:, t] = function(exo_vars, sem_samples, epsilon, t)

    return sem_samples
