import numpy as np
from src.utils.forecast import make_conditional_bernoulli


def fit_trans_mat(obs):
    unique_values, value_counts = np.unique(obs, return_counts=True, axis=0)
    n = len(unique_values)
    assert n % 2 == 0, "There are not an even number of unique rows."
    m = np.sqrt(len(unique_values))
    M = value_counts.reshape((m, m))
    row_sums = M.sum(axis=1, keepdims=True)
    return M / row_sums


def fit_sem_transition_functions(observational_samples, transfer_pairs: dict) -> dict:

    # Store function which concern t-1 --> t
    transition_functions = {}

    for input_vars in transfer_pairs.keys():
        # Transfer input
        if len(input_vars) > 1:
            raise NotImplementedError("Have to map tuples to integers.")
        else:
            in_var = input_vars[0].split("_")[0]
            in_time = int(input_vars[0].split("_")[1])
            # Transfer target
            output = transfer_pairs[input_vars]
            out_var = output.split("_")[0]
            out_time = int(output.split("_")[1])

            assert in_var == out_var, "Pairs: {} --> {}".format(in_var, out_var)
            assert in_time != out_time, "Pairs: {} --> {}".format(in_time, out_time)

        trans_mat = fit_trans_mat(observational_samples[out_var][:, [in_time, out_time]])

        # Store funcs in dict for later usage with given discrete forecasting model
        transition_functions[input_vars] = make_conditional_bernoulli(trans_mat)

    return transition_functions
