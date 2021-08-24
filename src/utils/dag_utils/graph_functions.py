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
            ranking.append("{{ rank=same; {} }} ".format(" ".join([item + "_{}".format(t) for item in nodes])))
    else:
        raise ValueError("Not a valid time-slice topology.")

    ## Confounding edges

    if confounder_info:
        confounders = []
        common_cause = 2 * "{}_{} -> {}_{} [style=dashed, color=red, constraint=false]; "
        for t in confounder_info.keys():
            if isinstance(confounder_info[t], list):
                raise NotImplementedError("Cannot yet have multiple UCs.")
            l1 = ["U", confounder_info[t][0], "U", confounder_info[t][-1]]
            l2 = 4 * [t]
            inserts = [val for pair in zip(l1, l2) for val in pair]
            confounders.append(common_cause.format(*inserts))
        time_slice_edges = "".join(time_slice_edges + confounders)
        #  Update ranking to take into account UC
        for t in confounder_info.keys():
            uc = " U_{}".format(t)
            idx = ranking[t].index("}")
            ranking[t] = ranking[t][:idx] + uc + ranking[t][idx:]
        ranking = "".join(ranking)
    else:
        time_slice_edges = "".join(time_slice_edges)
        ranking = "".join(ranking)

    ## Temporal edges

    temporal_edges = []
    if topology == "independent":
        node_count += 1
        nodes += [target_node]

    connections = node_count * "{}_{} -> {}_{}; "
    for t in range(stop_time):
        edge_pairs = repeat(nodes, 2).tolist()
        temporal_idx = node_count * [t, t + 1]
        iters = [iter(edge_pairs), iter(temporal_idx)]
        inserts = list(chain(map(next, cycle(iters)), *iters))
        temporal_edges.append(connections.format(*inserts))

    temporal_edges = "".join(temporal_edges)

    graph = "digraph {{ rankdir=LR; ranksep=1.1; {} {} {} }}".format(time_slice_edges, temporal_edges, ranking)

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
        ninfo = [node_information[node.split("_")[0]] for node in G.nodes]
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


def make_time_slice_causal_diagrams(sub_graphs: list, node_info: dict, confounder_info: dict) -> list:
    T = len(sub_graphs)
    sub_causal_diagrams = []

    for t in range(T):
        edges = [e[:-1] for e in sub_graphs[t].edges]
        directed_edges = [edge for edge in edges if all(sub_graphs[t].nodes[v]["type"] != "confounder" for v in edge)]
        directed_edges = [tuple([v.split("_")[0] for v in edge]) for edge in directed_edges]
        variables = set(
            [
                node.split("_")[0]
                for node in sub_graphs[0].nodes
                if node_info[node.split("_")[0]]["type"] != "confounder"
            ]
        )

        # Unobserved confounders here
        # TODO: currently only allow for ONE confounder per time-slice
        bi_edges = frozenset()
        if t in confounder_info.keys():
            bi_edges = [confounder_info[t] + ("U_{}".format(t),)]

        #  Set causal diagrams for this sub-graph
        sub_causal_diagrams.append(
            CausalDiagram(variables=variables, directed_edges=directed_edges, bidirected_edges=bi_edges)
        )

    return sub_causal_diagrams
