from collections import OrderedDict
import numpy as np


def sample_sem(
    sem, timesteps: int, sample_size: int, epsilon=None, seed=None,
):
    """
    Draw sample(s) from given structural equation model.

    Returns
    -------
    dict
        Sequential sample(s) from SEM.
    """

    SEM = sem(sample_size=sample_size)
    static = SEM.static()
    dynamic = SEM.dynamic()

    assert static.keys() == dynamic.keys()

    if seed:
        np.random.seed(seed)

    if epsilon:
        assert epsilon.shape == (sample_size, timesteps)

    # Pre-allocate the sample container
    s = OrderedDict([(k, timesteps * [None]) for k in static.keys()])

    for t in range(timesteps):
        model = static if t == 0 else dynamic
        for var, function in model.items():
            s[var][t] = function(s, t, epsilon)

    #  Convert each key from a list to 2D array
    for key in s.keys():
        if len(s[key][0].shape) == 1:
            s[key] = np.array(s[key]).T
        else:
            s[key] = np.hstack(s[key])

    return s
