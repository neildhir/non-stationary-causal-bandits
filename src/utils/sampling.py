import numpy as np
from collections import OrderedDict


def sample_sem(
    sem, background_samples: dict, timesteps: int, sample_size: int, seed=None,
):
    """
    Draw sample(s) from given structural equation model under steady-state condition (i.e. we do not fix any nodes as we walk along the graph).

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

    # Pre-allocate the sample container
    empty_samples = OrderedDict({k: np.empty((sample_size, timesteps), dtype="int") for k in static.keys()})

    # Note that samples is pre-allocated above and the background samples are pre-drawn as well
    samples = empty_samples | background_samples

    for t in range(timesteps):
        model = static if t == 0 else dynamic
        for var, function in model.items():
            samples[var][:, t] = function(samples, t)

    return samples
