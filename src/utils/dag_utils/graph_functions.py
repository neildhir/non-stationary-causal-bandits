from graphviz import Source
from numpy import repeat
from itertools import cycle, chain
from networkx import nx_agraph
import pygraphviz


def make_graphical_model(
    start_time, stop_time, topology, target_node, node_information, unobserved_confounder_info: None, verbose=False,
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

    if unobserved_confounder_info:
        confounders = []
        common_cause = "{}_{} <-- {}_{} --> {}_{}; "
        for t in unobserved_confounder_info.keys():
            l1 = [unobserved_confounder_info[t][0], "U", unobserved_confounder_info[t][-1]]
            l2 = 3 * [t]
            inserts = [val for pair in zip(l1, l2) for val in pair]
            confounders.append(common_cause.format(*inserts))
        # confounders = "".join(ucs)
        time_slice_edges = "".join(time_slice_edges + confounders)
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

    graph = "digraph {{ rankdir=LR; {} {} {} }}".format(time_slice_edges, temporal_edges, ranking)

    if verbose:
        return Source(graph)
    else:
        return graph


def make_networkx_object(graph):
    return nx_agraph.from_agraph(pygraphviz.AGraph(graph.source))
