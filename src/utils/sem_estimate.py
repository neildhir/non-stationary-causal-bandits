from collections import OrderedDict


def make_sem_hat() -> classmethod:
    class DynamicIVCDHat:
        @staticmethod
        def static() -> OrderedDict:
            return OrderedDict(
                {
                    "Z": lambda v: v["U_Z"],
                    "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"],
                    "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"],
                }
            )

        @staticmethod
        def dynamic(clamped: dict = None) -> OrderedDict:
            return OrderedDict(
                {
                    "Z": lambda v: v["U_Z"] if clamped["Z"] is None else v["U_Z"] ^ clamped["Z"],
                    "X": lambda v: v["U_X"] ^ v["U_XY"] ^ v["Z"]
                    if clamped["X"] is None
                    else v["U_X"] ^ v["U_XY"] ^ v["Z"] ^ clamped["X"],
                    "Y": lambda v: 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"]
                    if clamped["Y"] is None
                    else 1 ^ v["U_Y"] ^ v["U_XY"] ^ v["X"] ^ clamped["Y"],
                }
            )

    return DynamicIVCDHat
