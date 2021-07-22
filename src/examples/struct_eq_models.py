from collections import OrderedDict
from os import stat
from networkx.algorithms.similarity import optimize_graph_edit_distance
import numpy as np


class DynamicIVCD:
    """
    Dynamic Instrumental Variable Structural equation model.

    Adapted from from task 3 (experimental section): Structural Causal Bandits: Where to Intervene?

    Notes
    -----
    1. ^ is the XOR operator used to model binary interactions
    2. The discrete white noise definition is quite similar to the continuous noise definition, meaning that it has mean zero (constant), and its variance is also constant (nonzero), and there's no autocorrelation
    3. U_XY is the confounding variable between X and Y
    4. We can bolt-on transition functions too to make the problem even harder
    """

    def __init__(self):
        """
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
        time index: t
        sample: s
        noise: e
        """

        # TODO: add sample size so here so we can parallelise the sampler
        # TODO: need to write a method which estimates this (i.e. in real life we do not have access to the SEM)

        return {
            "Z": lambda t, s, e: s["U_Z"][t] ^ e if e else s["U_Z"][t],
            "X": lambda t, s, e: s["U_X"][t] ^ s["U_XY"][t] ^ s["Z"][t] ^ e
            if e
            else s["U_X"][t] ^ s["U_XY"][t] ^ s["Z"][t],
            "Y": (
                lambda t, s, e: 1 ^ s["U_Y"][t] ^ s["U_XY"][t] ^ s["X"][t] ^ e
                if e  #  if e is passed as None then the else statement is invoked
                else 1 ^ s["U_Y"][t] ^ s["U_XY"][t] ^ s["X"][t]
            ),
        }

    @staticmethod
    def dynamic(assigned: dict) -> dict:
        """
        Parameters
        ----------
        assigned : dict
            Contains the value of the assigned variables in the global SCM, from the previous time-step (previous MAB problem)

        variable: V
        noise: e
        """
        return {
            # z_{-1} --> Z <-- U_Z
            "Z": lambda V, e: V["U_Z"] ^ assigned["Z"] ^ e if e else V["U_Z"] ^ assigned["Z"],
            # x_{-1} --> X <-- {U_X, U_XY, Z}
            "X": (
                # Clean
                lambda V, e: V["U_X"] ^ V["U_XY"] ^ V["Z"] ^ assigned["X"] ^ e
                if e
                # Noisy
                else V["U_X"] ^ V["U_XY"] ^ V["Z"] ^ assigned["X"]
            ),
            # y_{-1} --> Y <-- {U_Y, U_XY, X}
            "Y": (
                lambda V, e: 1 ^ V["U_Y"] ^ V["U_XY"] ^ V["X"] ^ assigned["Y"] ^ e
                if e
                else 1 ^ V["U_Y"] ^ V["U_XY"] ^ V["X"] ^ assigned["Y"]
            ),
        }

