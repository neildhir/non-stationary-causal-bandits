from collections import OrderedDict


class DynamicIVCD:
    """
    Dynamic Instrumental Variable Structural equation model.

    Adapted from from task 3 (experimental section): 'Structural Causal Bandits: Where to Intervene?'

    Note that the dictionary returned respects the causal ordering of each time-slice in the graph.

    Notes
    -----
    1. ^ is the XOR operator used to model binary interactions (remember that ^ (xor) is a bitwise operation)
    2. The discrete white noise definition is quite similar to the continuous noise definition, meaning that it has mean zero (constant), and its variance is also constant (nonzero), and there's no autocorrelation
    3. U_XY is the confounding variable between X and Y
    """

    @staticmethod
    def static() -> OrderedDict:
        """
        Parameters
        ----------
        v: SCM variables
        """
        return OrderedDict(
            {
                "Z": lambda v: v["U_Z"],
                "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"],
                "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"],
            }
        )

    @staticmethod
    def static_vec() -> OrderedDict:
        """
        Parameters
        ----------
        SCM variables: v
        noise: e
        time index: t
        """

        return OrderedDict(
            {
                "Z": lambda v, t: v["U_Z"][:, t],
                "X": lambda v, t: v["U_X"][:, t] ^ v["U_XY"][:, t] ^ v["Z"][:, t],
                "Y": lambda v, t: 1 ^ v["U_Y"][:, t] ^ v["U_XY"][:, t] ^ v["X"][:, t],
            }
        )

    @staticmethod
    def dynamic(clamped: dict = None) -> OrderedDict:
        """
        Parameters
        ----------
        clamped: clamped (fixed) variables from the previous time-step (type: dict)

        # Note that the time operator theorem from DCBO says that the previous target value y_{t-1} necessarily needs to be _added_ to the current value but here we are _not_ doing that. Currently not sure about the implications of that.

        Lambda function input parameters
        --------------------------------
        v: SCM variables (endogenous and exogenous)
        """

        return OrderedDict(
            {
                # f_Z(z_{t-1}) [the 'clamped' part if it exists, where f_Z is the transition function] --> Z <-- U_Z
                "Z": lambda v: v["U_Z"] if clamped["Z"] is None else v["U_Z"] ^ clamped["Z"],
                # f_X(x_{t-1}) --> X <-- {U_X, U_XY, Z}
                "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"]
                if clamped["X"] is None
                else v["U_X"] ^ v["U_XY"] ^ v["Z"] ^ clamped["X"],
                # f_Y(y_{t-1}) --> Y <-- {U_Y, U_XY, X}
                "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"]
                if clamped["Y"] is None
                else 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"] ^ clamped["Y"],
                #
            }
        )

    @staticmethod
    def dynamic_vec(clamped=None) -> dict:
        """
        Parameters
        ----------
        clamped: clamped variables from the previous time-step (type: dict), otherwise we are just sampling the system in its steady-state behaviour.

        Lambda function input parameters
        --------------------------------
        v: SCM variables (type: dict containing np.ndarrays -- one per var)
        t: time index (type: int)
        """
        return OrderedDict(
            {
                # z_{t-1} (the 'clamped' part if it exists) --> Z <-- U_Z
                "Z": (lambda v, t: v["U_Z"][:, t] ^ (v["Z"][:, t - 1] if clamped is None else clamped["Z"])),
                # x_{t-1} --> X <-- {U_X, U_XY, Z}
                "X": (
                    lambda v, t: v["U_X"][:, t]
                    ^ v["U_XY"][:, t]
                    ^ v["Z"][:, t]
                    ^ (v["X"][:, t - 1] if clamped is None else clamped["X"])
                ),
                # y_{t-1} --> Y <-- {U_Y, U_XY, X}
                "Y": (
                    lambda v, t: 1
                    ^ v["U_Y"][:, t]
                    ^ v["U_XY"][:, t]
                    ^ v["X"][:, t]
                    ^ (v["Y"][:, t - 1] if clamped is None else clamped["Y"])
                ),
            }
        )


class test:
    """
    Dynamic Instrumental Variable Structural equation model.

    Adapted from from task 3 (experimental section): 'Structural Causal Bandits: Where to Intervene?'

    Note that the dictionary returned respects the causal ordering of each time-slice in the graph.

    Notes
    -----
    1. ^ is the XOR operator used to model binary interactions (remember that ^ (xor) is a bitwise operation)
    2. The discrete white noise definition is quite similar to the continuous noise definition, meaning that it has mean zero (constant), and its variance is also constant (nonzero), and there's no autocorrelation
    3. U_XY is the confounding variable between X and Y
    """

    @staticmethod
    def static() -> OrderedDict:
        """
        Parameters
        ----------
        v: SCM variables
        """
        return OrderedDict(
            {
                "Z": lambda v: v["U_Z"],
                "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"],
                "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"],
            }
        )

    @staticmethod
    def dynamic(clamped: dict = None) -> OrderedDict:
        """
        Parameters
        ----------
        clamped: clamped (fixed) variables from the previous time-step (type: dict)

        # Note that the time operator theorem from DCBO says that the previous target value y_{t-1} necessarily needs to be _added_ to the current value but here we are _not_ doing that. Currently not sure about the implications of that.

        Lambda function input parameters
        --------------------------------
        v: SCM variables (endogenous and exogenous)
        """

        return OrderedDict(
            {
                # f_Z(z_{t-1}) [the 'clamped' part if it exists, where f_Z is the transition function] --> Z <-- U_Z
                "Z": lambda v: v["U_Z"] if clamped["Z"] is None else v["U_Z"] ^ clamped["Z"],
                # f_X(x_{t-1}) --> X <-- {U_X, U_XY, Z}
                "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"]
                if clamped["X"] is None
                else v["U_X"] ^ v["U_XY"] ^ v["Z"] ^ clamped["X"],
                # f_Y(y_{t-1}) --> Y <-- {U_Y, U_XY, X}
                "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"]
                if clamped["Y"] is None
                else 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"] ^ clamped["Y"],
                #
            }
        )

