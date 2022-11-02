import argparse
import itertools as it
import json
import logging
import networkx as nx
import pandas as pd
from pyproj import CRS, Transformer
import re
import requests
from tqdm import tqdm
from typing import Optional


BASE_URL = "https://api.tfl.gov.uk"
# These new stations may not have any passenger data because they were opened in 2021.
NEW_STATIONS = {
    "940GZZBPSUST",  # Battersea Power Station
    "940GZZNEUGST",  # Nine Elms
}
NAME_LOOKUP = {
    "Edgware Road (Bakerloo)": "Edgware Road (Bak)",
    "Edgware Road (Circle Line)": "Edgware Road (DIS)",
    "Hammersmith (H&C Line)": "Hammersmith (H&C)",
    "Hammersmith (Dist&Picc Line)": "Hammersmith (DIS)",
    "Heathrow Terminals 2 & 3": "Heathrow Terminals 123",
    "Paddington": "Paddington TfL",
    "Shepherd's Bush (Central)": "Shepherd's Bush",
}
LOGGER = logging.getLogger(__name__)


def get_and_parse(path, base_url=None, app_key=None, **params):
    """
    Get a http response, check for errors, and parse the result as JSON.

    Args:
        path: Remote path to fetch.
        base_url: Base url or domain to fetch from.
        app_key: TfL application key.
        **params: Additional parameters passed to the request.
    """
    base_url = base_url or BASE_URL
    if app_key:
        params.setdefault("app_key", app_key)
    response = requests.get(f"{base_url}/{path}", params)
    response.raise_for_status()
    return response.json()


def get_nodes(graph, *args, **kwargs):
    """
    Get nodes that satisfy all callable criteria and match keyword arguments exactly.

    Args:
        graph: Graph to get nodes from.
        *args: Callable criteria taking node data as input.
        **kwargs: Named attributes that must match exactly.
    """
    return {node: data for node, data in graph.nodes(data=True) if
            all(arg(data) for arg in args) and
            all(data[key] == value for key, value in kwargs.items())}


def merge_stations(graph: nx.Graph, keep: str, remove: str, **attrs) -> None:
    # Copy any edges to the node we keep.
    for node, other, data in graph.edges(remove, data=True):
        assert node == remove, "weird ordering"
        assert node != other, "this creates a self loop"
        edge_data = graph.get_edge_data(keep, other)
        if edge_data is None:
            graph.add_edge(keep, other, lines=set(data["lines"]))
        else:
            edge_data["lines"].update(data["lines"])

    # Update attributes of the node we keep and remove the other.
    graph.nodes[keep].update(attrs)
    graph.remove_node(remove)


def encode_set(x: set) -> list:
    """
    Convert a set to a list for JSON encoding.
    """
    if isinstance(x, set):
        return list(x)
    raise TypeError(type(x))


def __main__(args: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app_key", help="TfL app key (see https://api-portal.tfl.gov.uk/faq for "
                        "details)")
    parser.add_argument("annualized_entry_exit", help="Excel sheet of station entries and exits")
    parser.add_argument("output", help="JSON output file")
    args = parser.parse_args()

    if not args.app_key:
        LOGGER.warning("TfL app key is not available as environment variable `TFL_APP_KEY`; "
                       "fetching information may be slow or fail")

    # Get all lines identifiers and their metadata.
    line_ids = [line["id"] for line in get_and_parse("Line/Mode/tube/Route")]
    print(f"found {len(line_ids)} lines: {', '.join(line_ids)}")
    lines = [get_and_parse(f'Line/{line_id}/Route/Sequence/all') for line_id in tqdm(line_ids)]

    # Add all stations and connections to the graph.
    graph = nx.Graph()
    for line in lines:
        line_id = line["lineId"]
        for stop_point_sequence in line["stopPointSequences"]:
            for stop_points in it.pairwise(stop_point_sequence["stopPoint"]):
                stop_ids = []
                for station in stop_points:
                    zones = [int(zone) for zone in re.split(r"[/+]", station["zone"])]
                    graph.add_node(station["id"], name=station["name"], zone=min(zones),
                                   lat=station["lat"], lon=station["lon"], zones=zones)
                    stop_ids.append(station["id"])
                u, v = stop_ids
                edge_data = graph.get_edge_data(u, v)
                if edge_data is None:
                    graph.add_edge(u, v, lines={line_id})
                elif line_id not in edge_data:
                    edge_data["lines"].add(line_id)

    # Project to the British National Grid.
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(27700))
    for node, data in graph.nodes(data=True):
        data["x"], data["y"] = transformer.transform(data["lat"], data["lon"])

    # Merge Bank and Monument as well as the two Paddington Underground stations because that's the
    # level at which entry and exit data are available.
    bank, = get_nodes(graph, name="Bank Underground Station")
    monument, = get_nodes(graph, name="Monument Underground Station")
    merge_stations(graph, bank, monument, name="Bank and Monument Underground Station")

    paddington, = get_nodes(graph, name="Paddington Underground Station")
    paddington_hc, = get_nodes(graph, name="Paddington (H&C Line)-Underground")
    merge_stations(graph, paddington, paddington_hc, name="Bank and Monument Underground Station")

    # Load the data and filter to only London Underground (LU).
    ee = pd.read_excel(args.annualized_entry_exit, sheet_name="Annualised", skiprows=6)
    ee = ee[ee.Mode == "LU"]
    # Remove the "LU" suffix where given.
    ee.Station = ee.Station.str.removesuffix(" LU")
    # Sanity check that there are no conflicts and then use the station name as the index.
    assert len(ee) == ee.Station.nunique(), "there are duplicate stations"
    ee = ee.set_index("Station")

    # Transfer the entry and exit volumes to the graph.
    for node, data in graph.nodes(data=True):
        # Skip new stations for which we know data to be missing.
        if node in NEW_STATIONS:
            data["entries"] = data["exits"] = None
            continue

        # Wrangle the names to match against the spreadsheet.
        name = data["name"].removesuffix(" Underground Station")
        name = NAME_LOOKUP.get(name, name)

        # Update the data or complain if it's missing.
        try:
            row = ee.loc[name]
            data["entries"] = int(row.entries)
            data["exits"] = int(row.exits)
        except KeyError:
            LOGGER.warning(f"missing entry and exit data for {name} ({node})")

    # Save the results.
    data = {
        "nodes": dict(graph.nodes(data=True)),
        "edges": list(graph.edges(data=True)),
    }

    with open(args.output, "w") as fp:
        json.dump(data, fp, indent=4, default=encode_set)

    print(f"dumped graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges "
          f"to `{args.output}`")


if __name__ == "__main__":
    __main__()
