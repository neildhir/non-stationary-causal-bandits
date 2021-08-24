from npsem.utils import rand_bw, seeded
from examples.struct_eq_models import DynamicIVCD
from utils.dag_utils.graph_functions import make_graphical_model, make_networkx_object


def setup_DynamicIVCD():

    with seeded(seed=0):
        mu1 = {
            "U_X": rand_bw(0.01, 0.2, precision=2),
            "U_Y": rand_bw(0.01, 0.2, precision=2),
            "U_Z": rand_bw(0.01, 0.99, precision=2),
            "U_XY": rand_bw(0.4, 0.6, precision=2),
        }
        node_info = {
            "Z": {"type": "manipulative", "domain": (0, 1)},
            "X": {"type": "manipulative", "domain": (0, 1)},
            "Y": {"type": "manipulative", "domain": (-1, 1)},
            "U": {"type": "confounder", "domain": None},
        }
        # Constructor for adding unobserved confounder to graph
        uc_constructor = {0: ("X", "Y"), 1: ("X", "Y"), 2: ("X", "Y")}

        T = 3
        graph_view = make_graphical_model(
            0,
            T - 1,
            topology="dependent",
            target_node="Y",
            node_information=node_info,
            confounder_info=uc_constructor,
            verbose=False,
        )

        G = make_networkx_object(graph_view, node_info)

        return {
            "G": G,
            "SEM": DynamicIVCD,
            "mu1": mu1, # Reward distribution
            "node_info": node_info,
            "confounder_info": uc_constructor,
            "base_target_variable": "Y",
            "horizon": 10,
            "n_trials": 2,
            "n_jobs": 2,
        }
