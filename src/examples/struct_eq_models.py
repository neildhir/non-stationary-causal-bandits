from collections import OrderedDict
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
    """

    def __init__(self, sample_size):
        transmatZ = {0: [0.5, 0.5], 1: [0.35, 0.65]}
        transmatX = {0: [0.1, 0.9], 1: [0.8, 0.2]}
        transmatY = {0: [0.75, 0.25], 1: [0.3, 0.7]}
        self.trans_funcZ = np.vectorize(lambda state: np.random.choice(2, 1, p=transmatZ[state]))
        self.trans_funcX = np.vectorize(lambda state: np.random.choice(2, 1, p=transmatX[state]))
        self.trans_funcY = np.vectorize(lambda state: np.random.choice(2, 1, p=transmatY[state]))
        self.sample_size = sample_size

    def static(self):
        """
        time index: t
        sample: s
        noise: e
        """

        # TODO: add sample size so here so we can parallelise the sampler
        Z = lambda t, s, e: s["U_Z"][t] ^ e if e else s["U_Z"][t]
        X = lambda t, s, e: s["U_X"][t] ^ s["U_XY"][t] ^ s["Z"][t] ^ e if e else s["U_X"][t] ^ s["U_XY"][t] ^ s["Z"][t]
        Y = (
            lambda t, s, e: 1 ^ s["U_Y"][t] ^ s["U_XY"][t] ^ s["X"][t] ^ e
            if e  #  if e is passed as None then the else statement is invoked
            else 1 ^ s["U_Y"][t] ^ s["U_XY"][t] ^ s["X"][t]
        )
        return OrderedDict([("Z", Z), ("X", X), ("Y", Y)])

    def dynamic(self):
        """
        time index: t
        sample: s
        noise: e
        """
        Z = (
            lambda t, s, e: s["U_Z"][t] ^ self.trans_funcZ(s["Z"][t - 1]) ^ e
            if e
            else s["U_Z"][t] ^ self.trans_funcZ(s["Z"][t - 1])
        )
        X = (
            lambda t, s, e: s["U_X"][t] ^ s["U_XY"][t] ^ s["Z"][t] ^ self.trans_funcX(s["X"][t - 1]) ^ e
            if e
            else s["U_X"][t] ^ s["U_XY"][t] ^ s["Z"][t] ^ self.trans_funcX(s["X"][t - 1])
        )
        Y = (
            lambda t, s, e: 1 ^ s["U_Y"][t] ^ s["U_XY"][t] ^ s["X"][t] ^ self.trans_funcY(s["Y"][t - 1]) ^ e
            if e
            else 1 ^ s["U_Y"][t] ^ s["U_XY"][t] ^ s["X"][t] ^ self.trans_funcY(s["Y"][t - 1])
        )
        return OrderedDict([("Z", Z), ("X", X), ("Y", Y)])
