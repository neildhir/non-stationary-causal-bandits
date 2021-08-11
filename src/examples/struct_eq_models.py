from collections import OrderedDict

# from os import stat
# from networkx.algorithms.similarity import optimize_graph_edit_distance
import numpy as np


class DynamicIVCD:
    """
    Dynamic Instrumental Variable Structural equation model.

    Adapted from from task 3 (experimental section): Structural Causal Bandits: Where to Intervene?

    Note that the dictionary returned respects the causal ordering of each time-slice in the graph.

    Notes
    -----
    1. ^ is the XOR operator used to model binary interactions
    2. The discrete white noise definition is quite similar to the continuous noise definition, meaning that it has mean zero (constant), and its variance is also constant (nonzero), and there's no autocorrelation
    3. U_XY is the confounding variable between X and Y
    4. We can bolt-on transition functions too to make the problem even harder
    """

    def __init__(self):
        """
        Not currently used. Will be used if we transition to a MDP/POMDP scenario and have requirements for transition functions.
        """
        transmatZ = {0: [0.5, 0.5], 1: [0.35, 0.65]}
        transmatX = {0: [0.1, 0.9], 1: [0.8, 0.2]}
        transmatY = {0: [0.75, 0.25], 1: [0.3, 0.7]}
        # Usage: self.trans_funcZ(s["Z"][t - 1])
        self.trans_funcZ = np.vectorize(lambda state: np.random.choice(2, 1, p=transmatZ[state]))
        self.trans_funcX = np.vectorize(lambda state: np.random.choice(2, 1, p=transmatX[state]))
        self.trans_funcY = np.vectorize(lambda state: np.random.choice(2, 1, p=transmatY[state]))

    @staticmethod
    def static() -> OrderedDict:
        """
        variable: v
        noise: e
        """

        # TODO: add sample size so here so we can parallelise the sampler
        # TODO: need to write a method which estimates this (i.e. in real life we do not have access to the SEM)

        return {
            "Z": lambda u, v, e, t: u["U_Z"][t] ^ e if e else u["U_Z"][t],
            "X": lambda u, v, e, t: u["U_X"][t] ^ u["U_XY"][t] ^ v["Z"][t] ^ e
            if e
            else u["U_X"][t] ^ u["U_XY"][t] ^ v["Z"][t],
            "Y": (
                lambda u, v, e, t: 1 ^ u["U_Y"][t] ^ u["U_XY"][t] ^ v["X"][t] ^ e
                if e  #  if e iv pavved av None then the elve vtatement iv invoked
                else 1 ^ u["U_Y"][t] ^ u["U_XY"][t] ^ v["X"][t]
            ),
        }

    @staticmethod
    def dynamic(clamped: dict) -> dict:
        """
        Parameters
        ----------
        clamped: dict
            Contains the value of the assigned variables in the global SCM, from the previous time-step (previous MAB problem)
        variable: v
        noise: e

        Note
        ----
        Small-caps in the comment means the clamped variable i.e. 'clamped["X"][t] == x_{t-1}' i.e. the X node at t-1 is clamped to value x_{t-1}.
        """
        return {
            # z_{t-1} --> Z <-- U_Z
            "Z": lambda u, v, e, t: u["U_Z"][t] ^ v["Z"][t - 1] ^ e if e else u["U_Z"][t] ^ v["Z"][t - 1],
            # x_{t-1} --> X <-- {U_X, U_XY, Z}
            "X": (
                # Clean
                lambda u, v, e, t: u["U_X"][t] ^ u["U_XY"][t] ^ v["Z"][t] ^ v["X"][t - 1] ^ e
                if e
                # Noisy
                else u["U_X"][t] ^ u["U_XY"][t] ^ v["Z"][t] ^ v["X"][t - 1]
            ),
            # y_{t-1} --> Y <-- {U_Y, U_XY, X}
            "Y": (
                lambda u, v, e, t: 1 ^ u["U_Y"][t] ^ u["U_XY"][t] ^ v["X"][t] ^ v["Y"][t - 1] ^ e
                if e
                else 1 ^ u["U_Y"][t] ^ u["U_XY"][t] ^ v["X"][t] ^ v["Y"][t - 1]
            ),
        }

