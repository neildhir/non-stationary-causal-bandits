import numpy as np
from collections import OrderedDict


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
    def static() -> dict:
        """
        Parameters
        ----------
        SCM variables: v
        noise: e
        time index: t
        """

        # TODO: need to write a method which estimates this (i.e. in real life we do not have access to the SEM)

        return OrderedDict(
            {
                "Z": lambda v, t: v["U_Z"][:, t],
                "X": lambda v, t: v["U_X"][:, t] ^ v["U_XY"][:, t] ^ v["Z"][:, t],
                "Y": lambda v, t: 1 ^ v["U_Y"][:, t] ^ v["U_XY"][:, t] ^ v["X"][:, t],
            }
        )

    @staticmethod
    def dynamic(clamped=None) -> dict:
        """
        Parameters
        ----------
        clamped: clamped variables from the previous time-step (type: dict), otherwise we are just sampling the system in its steady-state behaviour.

        Lambda function input parameters
        --------------------------------
        v: SCM variables (type: dict containing np.ndarraysa -- one per var)
        t: time index (type: int)
        """
        return OrderedDict(
            {
                # z_{t-1} (the 'clamped' part) --> Z <-- U_Z
                # TODO:if clamped is None the v["Z"][:,t-1] needs to be sampled? More importantly because v["Z"][:,t-1] is a PMF it needs to be sampled each time this functional SEM is called.
                "Z": (lambda v, t: v["U_Z"][:, t] ^ (v["Z"][:, t - 1] if clamped is None else clamped["Z"])),
                # x_{t-1} --> X <-- {U_X, U_XY, Z}
                "X": (
                    lambda v, t: v["U_X"][:, t]
                    ^ v["U_XY"][:, t]
                    ^ v["Z"][:, t]
                    ^ (v["X"][:, t - 1] if clamped is None else clamped["X"])
                ),
                # TODO: note that the time operator theorem from DCBO says that the previous target value y_{t-1} necessarily needs to be _added_ to the current value but here we are _not_ doing that. Currently not sure about the implications of that.
                # y_{t-1} --> Y <-- {U_Y, U_XY, X}
                "Y": (
                    lambda v, t: 1
                    ^ v["U_Y"][:, t]  #  Remember that ^ (xor) is a bitwise operation
                    ^ v["U_XY"][:, t]
                    ^ v["X"][:, t]
                    ^ (v["Y"][:, t - 1] if clamped is None else clamped["Y"])
                ),
            }
        )

