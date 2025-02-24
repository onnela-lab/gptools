import argparse
from gptools.util.graph import check_edge_index, graph_to_edge_index
from gptools.util import encode_one_hot
import json
import networkx as nx
import numpy as np
from typing import List, Optional


def get_node_attribute(graph: nx.Graph, key: str) -> np.asarray:
    """
    Get a node attribute with consistent order.
    """
    return np.asarray([data[key] for _, data in sorted(graph.nodes(data=True))])


def __main__(args: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", help="path to graph json file")
    parser.add_argument("data", help="path to Stan data json file")
    args = parser.parse_args(args)

    with open(args.graph) as fp:
        data = json.load(fp)

    graph = nx.Graph()
    graph.add_nodes_from(data["nodes"].items())
    graph.add_edges_from(data["edges"])

    # Remove the new stations that don't have data yet (should just be the two new Northern Line
    # stations).
    stations_to_remove = [
        node for node, data in graph.nodes(data=True) if data["entries"] is None
    ]
    assert len(stations_to_remove) == 2
    graph.remove_nodes_from(stations_to_remove)
    # Remove Kensington Olympia because it's hardly used in regular transit.
    graph.remove_node("940GZZLUKOY")
    print(
        f"loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    # Get passenger numbers and station locations.
    y = get_node_attribute(graph, "entries") + get_node_attribute(graph, "exits")
    X = (
        np.transpose([get_node_attribute(graph, "x"), get_node_attribute(graph, "y")])
        / 1000
    )
    X = X - np.mean(X, axis=0)

    # One-hot encode nodes and zones.
    max_zone = 6
    max_degree = 5
    zones = get_node_attribute(graph, "zone")
    one_hot_zones = encode_one_hot(zones.clip(max=max_zone) - 1)
    degrees = np.asarray([graph.degree[node] for node in sorted(graph)])
    one_hot_degrees = encode_one_hot(degrees.clip(max=max_degree) - 1)

    # Convert the graph to an edge index to pass to Stan and retain an inverse mapping so we can
    # look up stations again.
    edge_index = graph_to_edge_index(graph)
    check_edge_index(edge_index)

    data = {
        "num_stations": graph.number_of_nodes(),
        "num_edges": edge_index.shape[1],
        "edge_index": edge_index,
        "one_hot_zones": one_hot_zones,
        "num_zones": one_hot_zones.shape[1],
        "one_hot_degrees": one_hot_degrees,
        "num_degrees": one_hot_degrees.shape[1],
        "passengers": y,
        "station_locations": X,
    }
    # Cast any numpy arrays to lists to ensure they are json-serializable.
    data = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }

    with open(args.data, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    __main__()
