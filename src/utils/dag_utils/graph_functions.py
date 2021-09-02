from graphviz import Source
from networkx.classes.multidigraph import MultiDiGraph
from numpy import repeat
from itertools import cycle, chain
from networkx import nx_agraph, set_node_attributes
import pygraphviz
from npsem.model import CausalDiagram
from typing import Union


def make_graphical_model(
    start_time: int,
    stop_time: int,
    topology: str,
    target_node: int,
    node_information: dict,
    confounder_info: None,
    verbose=False,
):

    """
    Generic temporal Bayesian network with two types of connections.

    Parameters
    ----------
    start : int
        Index of first time-step
    stop : int
        Index of the last time-step
    topology: str, optional
        Choice of (spatial, i.e. per time-slice) topology
    target_node: str, optional
        If we are using a independent spatial topology then we need to specify the target node
    verbose : bool, optional
        To print the graph or not.

    Returns
    -------
    str
        Returns the DOT format of the graph

    Raises
    ------
    ValueError
        If an unknown topology is passed as argument.
    """

    assert start_time <= stop_time
    assert topology in ["dependent", "independent"]
    T = stop_time + 1

    # Get manipulative nodes
    nodes = [key for key in node_information.keys() if node_information[key]["type"] != "confounder"]

    if topology == "independent":
        assert target_node is not None
        assert isinstance(target_node, str)

    ## Time-slice edges

    time_slice_edges = []
    ranking = []
    # Check if target node is in the list of nodes, and if so remove it
    if topology == "independent":
        if target_node in nodes:
            nodes.remove(target_node)
        node_count = len(nodes)
        assert target_node not in nodes
        connections = node_count * "{}_{} -> {}_{}; "
        edge_pairs = list(sum([(item, target_node) for item in nodes], ()))
    else:
        node_count = len(nodes)
        connections = (node_count - 1) * "{}_{} -> {}_{}; "
        edge_pairs = [item for pair in list(zip(nodes, nodes[1:])) for item in pair]

    pair_count = len(edge_pairs)

    if topology == "independent":
        # X_0 --> Y_0; Z_0 --> Y_0
        all_nodes = nodes + [target_node]
        for t in range(start_time, stop_time + 1):
            space_idx = pair_count * [t]
            iters = [iter(edge_pairs), iter(space_idx)]
            inserts = list(chain(map(next, cycle(iters)), *iters))
            time_slice_edges.append(connections.format(*inserts))
            ranking.append("{{ rank=same; {} }} ".format(" ".join([item + "_{}".format(t) for item in all_nodes])))
    elif topology == "dependent":
        # X_0 --> Z_0; Z_0 --> Y_0
        for t in range(start_time, stop_time + 1):
            space_idx = pair_count * [t]
            iters = [iter(edge_pairs), iter(space_idx)]
            inserts = list(chain(map(next, cycle(iters)), *iters))
            time_slice_edges.append(connections.format(*inserts))
            ranking.append("{{ rank=same; {} }}".format(" ".join([item + "_{}".format(t) for item in nodes])))
    else:
        raise ValueError("Not a valid time-slice topology.")

    ## Background edges (noise/unobserved-factors variables and other stuff)

    background_edges = []
    connections = node_count * "{} -> {} [style=dashed]; "
    for t in range(start_time, T):
        main_node_info = [val for pair in zip(nodes, T * [str(t)]) for val in pair]
        main_nodes = ["_".join(x) for x in zip(main_node_info[0::2], main_node_info[1::2])]
        # Background variables (noise and so on)
        background_nodes = ["U_{}".format(i) for i in main_nodes]
        # background_nodes = ["U_{}".format(i) for i in nodes]
        background_edges.append(
            connections.format(*[val for pair in zip(background_nodes, main_nodes) for val in pair])
        )
        idx = ranking[t].index("}")
        ranking[t] = ranking[t][:idx] + " ".join(background_nodes) + ranking[t][idx:]

    background_edges = "".join(background_edges)

    ## Confounding edges

    if confounder_info:
        confounders = []
        common_cause = 2 * "{}_{} -> {}_{} [style=dashed, color=red, penwidth = 2, constraint=false]; "
        for t in confounder_info.keys():
            if isinstance(confounder_info[t], list):
                raise NotImplementedError("Cannot yet have multiple UCs.")
            l1 = [
                "U_{}".format("".join(confounder_info[t])),
                confounder_info[t][0],
                "U_{}".format("".join(confounder_info[t])),
                confounder_info[t][-1],
            ]
            l2 = 4 * [t]
            inserts = [val for pair in zip(l1, l2) for val in pair]
            confounders.append(common_cause.format(*inserts))
        time_slice_edges = "".join(time_slice_edges + confounders)
        #  Update ranking to take into account UC
        for t in confounder_info.keys():
            uc = " U_{}_{}".format("".join(confounder_info[t]), t)
            idx = ranking[t].index("}")
            ranking[t] = ranking[t][:idx] + uc + ranking[t][idx:]
        ranking = "".join(ranking)
    else:
        time_slice_edges = "".join(time_slice_edges)
        ranking = "".join(ranking)

    ## Time transition edges

    transition_edges = []
    if topology == "independent":
        node_count += 1
        nodes += [target_node]

    connections = node_count * "{}_{} -> {}_{}; "
    for t in range(stop_time):
        edge_pairs = repeat(nodes, 2).tolist()
        temporal_idx = node_count * [t, t + 1]
        iters = [iter(edge_pairs), iter(temporal_idx)]
        inserts = list(chain(map(next, cycle(iters)), *iters))
        transition_edges.append(connections.format(*inserts))

    transition_edges = "".join(transition_edges)

    graph = "digraph {{ rankdir=LR; ranksep=1.1; {} {} {} {}}}".format(
        time_slice_edges, background_edges, transition_edges, ranking
    )

    if verbose:
        return Source(graph)
    else:
        return graph


def make_networkx_object(graph: Union[str, MultiDiGraph], node_information: dict = None) -> MultiDiGraph:

    if isinstance(graph, str):
        G = nx_agraph.from_agraph(pygraphviz.AGraph(graph))
    else:
        G = nx_agraph.from_agraph(pygraphviz.AGraph(graph.source))

    if node_information:
        #  Sets what type of node each node is (manipulative, confounders, non-manipuatlive)
        ninfo = [node_information["_".join(node.split("_")[:-1])] for node in G.nodes]
        attrs = dict(zip(G.nodes, ninfo))
        set_node_attributes(G, attrs)
        return G
    else:
        return G


def get_time_slice_sub_graphs(G, T: int) -> list:
    sub_graphs = []
    for g in [[node for node in G.nodes if node.split("_")[-1] == str(t)] for t in range(T)]:
        sub_graphs.append(G.subgraph(g))
    return sub_graphs


def make_time_slice_causal_diagrams(sub_graphs: list, confounder_info: dict) -> list:
    T = len(sub_graphs)
    sub_causal_diagrams = []
    time_strip = lambda node: "_".join(node.split("_")[:-1])

    for t in range(T):
        edges = [e[:-1] for e in sub_graphs[t].edges]
        directed_edges = [edge for edge in edges if edge[0][0] != "U" and edge[-1][0] != "U"]
        directed_edges = [tuple([time_strip(v) for v in edge]) for edge in directed_edges]
        # directed_edges = [tuple([v.split("_")[0] for v in edge]) for edge in directed_edges]
        variables = set([node for node in [time_strip(node) for node in sub_graphs[t].nodes] if node[0] != "U"])

        # Unobserved confounders here
        bidirectional_edges = frozenset()
        if t in confounder_info.keys():
            # XXX: currently only allow for ONE confounder per time-slice
            bidirectional_edges = [confounder_info[t] + ("U_{}".format("".join(confounder_info[t])),)]

        #  Set causal diagrams for this sub-graph
        sub_causal_diagrams.append(
            CausalDiagram(variables=variables, directed_edges=directed_edges, bidirected_edges=bidirectional_edges)
        )

    return sub_causal_diagrams


def main():
    node_info = {
        # Endogenous
        "Z": {"type": "manipulative", "domain": (0, 1)},
        "X": {"type": "manipulative", "domain": (0, 1)},
        "Y": {"type": "manipulative", "domain": (0, 1)},
        # Exogenous / background conditions
        "U_Z": {"type": "background", "domain": (0, 1)},
        "U_X": {"type": "background", "domain": (0, 1)},
        "U_Y": {"type": "background", "domain": (0, 1)},
        "U_XY": {"type": "confounder", "domain": (0, 1)},
    }
    conf_info = {0: ("X", "Y"), 1: ("X", "Y"), 2: ("X", "Y")}
    T = 3
    make_graphical_model(
        0,
        T - 1,
        topology="dependent",
        target_node="Y",
        node_information=node_info,
        confounder_info=conf_info,
        verbose=True,
    )


if __name__ == "__main__":
    main()
