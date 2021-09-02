import numpy as np


def fit_trans_mat(mat):
    """
    Fit the transition matrix to observational samples of the form:

    'Z': array([[0, 0, 1, 0, 1],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 0, 1, 1, 1]])

                where each row is a discrete (but not necessarily binary) time-series.

    Parameters
    ----------
    mat : numpy ndarray
        Observational samples (time-series) from a given node in the DAG.

    Returns
    -------
    numpy ndarray
        Corresponding transition matrix, estimated from all time-series in the passed sample.
    """
    assert isinstance(mat, np.ndarray)
    n = len(np.unique(mat))
    M = np.zeros((n, n))
    # Assumes that each time-series lives on the rows of samples
    for row in mat:
        for (i, j) in zip(row, row[1:]):
            # Count each transition pair
            M[i][j] += 1
    row_sums = M.sum(axis=1, keepdims=True)
    return M / row_sums
