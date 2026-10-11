"""Charts. Every function draws one figure, saves it under ``outputs/`` and returns the path."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .compatibility import compatibility_row
from .similarity import similar_players
from .store import PlayerStore, get_store

ROLE_COLORS = {
    "Creative Playmaker": "#FFD700", "Deep Playmaker": "#00BFFF", "Ball Winner": "#FF4D4D",
    "Box-to-Box": "#32CD32", "Creative Winger": "#FF69B4", "Wide Winger": "#BA55D3",
    "Inside Forward": "#FF8C00", "Poacher": "#FFA500", "False 9": "#9370DB",
    "Target Forward": "#8B4513", "Ball Playing Defender": "#20B2AA",
    "Defensive Defender": "#708090", "Goalkeeper": "#000000",
}
LINE_COLORS = {"DEF": "#d62728", "MID": "#2ca02c", "WIDE": "#1f77b4", "FWD": "#9467bd"}
LINE_OF = {"CB": "DEF", "RB": "DEF", "LB": "DEF", "CDM": "MID", "CM": "MID", "CAM": "MID",
           "RM": "WIDE", "LM": "WIDE", "RW": "WIDE", "LW": "WIDE", "ST": "FWD"}


def _save(name: str) -> str:
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    path = config.OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close("all")
    return str(path)


def _barh(labels, values, title, xlabel, footer, cmap):
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = cmap(np.linspace(0.25, 0.9, len(values)))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=1.0)
    ax.invert_yaxis()
    for bar in bars:
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.figtext(0.5, 0.005, footer, ha="center", fontsize=11, fontweight="bold")


def similarity_chart(name: str, top_n: int = 10, store: PlayerStore | None = None) -> str | None:
    store = store or get_store()
    results = similar_players(name, top_n, store=store)
    if not results:
        return None
    target = store.names[store.find(name)]
    _barh([r["player"] for r in results], [r["similarity"] for r in results],
          f"FootballIQ Similarity Analysis\nTop Similar Players to {target}", "Similarity Score",
          f"Closest Match: {results[0]['player']} ({results[0]['similarity']:.3f})", plt.cm.viridis)
    return _save("similarity_chart.png")


def role_chart(name: str, store: PlayerStore | None = None) -> str | None:
    store = store or get_store()
    i = store.find(name)
    if i is None:
        return None
    top = sorted(store.role_scores(i).items(), key=lambda x: -x[1])[:5]
    _barh([r for r, _ in top], [s for _, s in top],
          f"FootballIQ Role Analysis\n{store.names[i]}", "Role Probability",
          f"Primary Role: {store.roles[i]}", plt.cm.plasma)
    return _save("role_probability_chart.png")


def compatibility_heatmap(names: list[str], store: PlayerStore | None = None) -> str | None:
    import seaborn as sns
    store = store or get_store()
    idx = [i for i in (store.find(n) for n in names) if i is not None]
    if len(idx) < 2:
        return None
    matrix = np.array([compatibility_row(i, store)[idx] for i in idx])
    labels = [store.names[i] for i in idx]
    plt.figure(figsize=(11, 8.5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", linewidths=1, square=True,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Compatibility Score"})
    plt.title("FootballIQ Player Compatibility Matrix", fontsize=18, fontweight="bold")
    return _save("compatibility_heatmap.png")


def squad_network(names: list[str], threshold: float = 0.60, store: PlayerStore | None = None) -> str | None:
    import networkx as nx
    store = store or get_store()
    idx = [i for i in (store.find(n) for n in names) if i is not None]
    if len(idx) < 2:
        return None
    graph = nx.Graph()
    graph.add_nodes_from(store.names[i] for i in idx)
    for a, i in enumerate(idx):
        row = compatibility_row(i, store)
        for j in idx[a + 1:]:
            if row[j] >= threshold:
                graph.add_edge(store.names[i], store.names[j], weight=float(row[j]))
    role_of = {store.names[i]: store.roles[i] for i in idx}
    pos = nx.kamada_kawai_layout(graph) if graph.number_of_edges() else nx.circular_layout(graph)
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(graph, pos, node_size=[1800 + 700 * graph.degree(n) for n in graph],
                           node_color=[ROLE_COLORS.get(role_of[n], "#A9A9A9") for n in graph],
                           edgecolors="black", linewidths=2, alpha=0.95)
    nx.draw_networkx_labels(graph, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(graph, pos, width=[graph[u][v]["weight"] * 8 for u, v in graph.edges()], alpha=0.7)
    nx.draw_networkx_edge_labels(graph, pos, font_size=8, edge_labels={
        (u, v): f"{graph[u][v]['weight']:.2f}" for u, v in graph.edges()})
    present = [r for r in ROLE_COLORS if r in role_of.values()]
    plt.legend(handles=[plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=ROLE_COLORS[r],
                                   markersize=10, label=r) for r in present],
               bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.title("FootballIQ Squad Compatibility Network", fontsize=20, fontweight="bold")
    plt.figtext(0.5, 0.02, "Node size = connectivity | Edge width = compatibility | Colour = role",
                ha="center", fontsize=10)
    plt.axis("off")
    return _save("squad_network.png")


def embedding_map(highlight: list[str] | None = None, min_minutes: int = 900,
                  store: PlayerStore | None = None) -> str:
    """t-SNE map of the outfield embedding, coloured by positional line."""
    from sklearn.manifold import TSNE
    store = store or get_store()
    keep = store.outfield[store.minutes[store.outfield] >= min_minutes]
    z = np.array([store.embedding(i) for i in keep])
    xy = TSNE(n_components=2, random_state=config.SEED, perplexity=30, init="pca").fit_transform(z)
    lines = np.array([LINE_OF[store.positions[i]] for i in keep], dtype=object)
    plt.style.use("default")
    plt.figure(figsize=(14, 10))
    for line, color in LINE_COLORS.items():
        m = lines == line
        plt.scatter(xy[m, 0], xy[m, 1], c=color, s=22, alpha=0.6, label=line, linewidths=0)
    for name in highlight or ["Pedri", "Rodri", "Federico Valverde", "Jude Bellingham",
                              "Erling Haaland", "Cole Palmer", "Vinicius Júnior", "Virgil van Dijk"]:
        i = store.find(name)
        if i is not None and i in set(keep):
            k = int(np.flatnonzero(keep == i)[0])
            plt.annotate(store.names[i], xy[k], fontsize=10, fontweight="bold",
                         xytext=(5, 5), textcoords="offset points")
    plt.legend(title="Line", fontsize=10)
    plt.title(f"FootballIQ player embedding (t-SNE, players with {min_minutes}+ minutes)",
              fontsize=16, fontweight="bold")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.2)
    return _save("latent_space.png")
