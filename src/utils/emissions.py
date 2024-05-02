from copy import deepcopy
from networkx.classes.multidigraph import MultiDiGraph
from itertools import combinations
from numpy.core.shape_base import hstack
from src.utils.forecast import make_conditional_bernoulli
from src.utils.transitions import fit_trans_mat


def get_emission_pairs(dag: MultiDiGraph) -> dict:
    T = dag.total_time

    emissions = {t: [] for t in range(T)}
    for e in dag.edges:
        e0, e1 = e[0].split("_"), e[1].split("_")
        inn_time, out_time = e0[-1], e1[-1]
        # Emission edge
        if out_time == inn_time:
            emissions[int(out_time)].append(("_".join(e0[:-1]), "_".join(e1[:-1])))

    # Deal with input/outputs for confounders
    new_emissions = deepcopy(emissions)
    for t in range(T):
        for a, b in combinations(emissions[t], 2):
            if a[0] == b[0]:
                new_emissions[t].append((a[0], (b[1], a[1])))
                new_emissions[t].remove(a)
                new_emissions[t].remove(b)

    # Format so that connections can change across time (incl. addition and removal of confounders)
    emission_pairs = {t: {} for t in range(T)}
    for t in range(T):
        for pair in new_emissions[t]:
            if isinstance(pair[0], tuple):
                emission_pairs[t][pair[0]] = pair[1]
            else:
                emission_pairs[t][(pair[0],)] = pair[1]

    return emission_pairs


def fit_emission_functions(observational_samples, emission_pairs: dict) -> dict:

    emission_functions = {}
    T = len(emission_pairs)

    for t in range(T):
        for input_vars in emission_pairs[t].keys():
            # Transfer input
            if len(input_vars) > 1:
                raise NotImplementedError("Have to map tuples to integers.")
            else:
                #  Input
                xx = observational_samples[input_vars[0]][:, t].reshape(-1, 1)

                #  Output
                output = emission_pairs[t][input_vars]
                if isinstance(output, tuple):
                    #  We've got a confounder which needs dealing with separately
                    assert len(output) > 1, (input_vars[0], output)
                    # [Confounder]
                    for node in output:
                        #  Input
                        yy = observational_samples[node][:, t].reshape(-1, 1)
                        # input --> output
                        trans_mat = fit_trans_mat(hstack((xx, yy)))
                        #  P(out | U)
                        emission_functions[node + " | " + input_vars] = make_conditional_bernoulli(trans_mat)
                else:
                    #  One-to-one mapping
                    yy = observational_samples[output][:, t]

            trans_mat = fit_trans_mat(hstack((xx, yy)))
            # Store funcs in dict for later usage with given discrete forecasting model
            emission_functions[input_vars] = make_conditional_bernoulli(trans_mat)

    return emission_functions
