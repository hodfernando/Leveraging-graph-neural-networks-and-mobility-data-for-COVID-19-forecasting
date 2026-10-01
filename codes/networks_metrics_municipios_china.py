import json
import os
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
NETWORKS_DIR = PROJECT_DIR / "raw_data" / "China" / "networks"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 1e-12


def numeric_weight(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def prepare_graph(graph):
    if "weight" not in graph.edge_attributes():
        graph.es["weight"] = [1.0] * graph.ecount()
    else:
        graph.es["weight"] = [numeric_weight(w) for w in graph.es["weight"]]

    graph.es["inverse_weight"] = [
        1.0 / max(numeric_weight(w), EPSILON)
        for w in graph.es["weight"]
    ]
    return graph


def vertex_names(graph):
    if "name" in graph.vertex_attributes():
        return [str(name) for name in graph.vs["name"]]
    return [str(i) for i in range(graph.vcount())]


def safe_closeness(graph, mode, weights=None):
    try:
        values = graph.closeness(
            vertices=None,
            mode=mode,
            weights=weights,
            normalized=True,
        )
    except Exception:
        values = [0.0] * graph.vcount()
    return [0.0 if value is None or not np.isfinite(value) else float(value) for value in values]


def safe_betweenness(graph, directed=True, weights=None):
    try:
        values = graph.betweenness(
            vertices=None,
            directed=directed,
            weights=weights,
        )
    except Exception:
        values = [0.0] * graph.vcount()
    return [0.0 if value is None or not np.isfinite(value) else float(value) for value in values]


def safe_diameter(graph, weights=None):
    try:
        value = graph.diameter(directed=True, weights=weights)
        return float(value) if np.isfinite(value) else 0.0
    except Exception:
        return 0.0


def global_metrics(graph, mode):
    degree = np.asarray(graph.degree(mode=mode), dtype=float)
    strength = np.asarray(graph.strength(weights="weight", mode=mode), dtype=float)

    mean_degree = float(degree.mean()) if degree.size else 0.0
    heterogeneity = (
        float(np.mean((degree - mean_degree) ** 2) / (mean_degree ** 2))
        if mean_degree > 0
        else 0.0
    )

    return {
        "nodes": int(graph.vcount()),
        "edges": int(graph.ecount()),
        "degree_mean": float(degree.mean()) if degree.size else 0.0,
        "degree_max": float(degree.max()) if degree.size else 0.0,
        "strength_mean": float(strength.mean()) if strength.size else 0.0,
        "strength_max": float(strength.max()) if strength.size else 0.0,
        "heterogeneity": heterogeneity,
        "density": float(graph.density()),
        "diameter_weighted": safe_diameter(graph, "inverse_weight"),
    }


def municipality_metrics(graph, mode, direction_label, graph_file):
    names = vertex_names(graph)
    degree = graph.degree(mode=mode)
    strength = graph.strength(weights="weight", mode=mode)
    closeness = safe_closeness(graph, mode, weights=None)
    closeness_weighted = safe_closeness(graph, mode, weights="inverse_weight")
    betweenness = safe_betweenness(graph, directed=True, weights=None)
    betweenness_weighted = safe_betweenness(
        graph, directed=True, weights="inverse_weight"
    )

    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "municipality_id": index,
                "municipality_name": name,
                "direction": direction_label,
                "graph_file": graph_file,
                "degree": int(degree[index]),
                "strength": float(strength[index]),
                "closeness": float(closeness[index]),
                "closeness_weighted": float(closeness_weighted[index]),
                "betweenness": float(betweenness[index]),
                "betweenness_weighted": float(betweenness_weighted[index]),
            }
        )
    return rows


def select_graph_files():
    input_files = sorted(
        path for path in NETWORKS_DIR.iterdir()
        if path.is_file() and "baidu_in" in path.name
    )
    output_files = sorted(
        path for path in NETWORKS_DIR.iterdir()
        if path.is_file() and "baidu_out" in path.name
    )
    return input_files, output_files


def write_outputs(all_global, all_municipal):
    with open(RESULTS_DIR / "network_metrics_china_global.json", "w", encoding="utf-8") as file:
        json.dump(all_global, file, ensure_ascii=False, indent=2)

    with open(RESULTS_DIR / "network_metrics_china_municipalities.json", "w", encoding="utf-8") as file:
        json.dump(all_municipal, file, ensure_ascii=False, indent=2)

    municipal_df = pd.DataFrame(all_municipal)
    municipal_df.to_csv(
        RESULTS_DIR / "network_metrics_china_municipalities.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main():
    if not NETWORKS_DIR.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {NETWORKS_DIR}")

    input_files, output_files = select_graph_files()
    all_global = []
    all_municipal = []

    for graph_file in input_files:
        graph = prepare_graph(ig.Graph.Read_GraphML(str(graph_file)))
        all_global.append(
            {
                "graph_file": graph_file.name,
                "direction": "in",
                **global_metrics(graph, mode="in"),
            }
        )
        all_municipal.extend(
            municipality_metrics(graph, "in", "in", graph_file.name)
        )

    for graph_file in output_files:
        graph = prepare_graph(ig.Graph.Read_GraphML(str(graph_file)))
        all_global.append(
            {
                "graph_file": graph_file.name,
                "direction": "out",
                **global_metrics(graph, mode="out"),
            }
        )
        all_municipal.extend(
            municipality_metrics(graph, "out", "out", graph_file.name)
        )

    write_outputs(all_global, all_municipal)
    print(f"Métricas globais: {RESULTS_DIR / 'network_metrics_china_global.json'}")
    print(f"Métricas municipais: {RESULTS_DIR / 'network_metrics_china_municipalities.csv'}")
    print(f"Métricas municipais JSON: {RESULTS_DIR / 'network_metrics_china_municipalities.json'}")


if __name__ == "__main__":
    main()
