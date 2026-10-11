"""Save every chart to ``outputs/``.

Usage:
    python save_outputs.py --player "Pedri"
"""
import argparse

from footballiq import charts
from footballiq.compatibility import best_partners

DEFAULT_GROUP = ["Pedri", "Rodri", "Federico Valverde", "Jude Bellingham", "Cole Palmer", "Moisés Caicedo"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", default="Pedri", help="focal player for the charts")
    args = parser.parse_args()

    partners = [r["player"] for r in best_partners(args.player, 5)]
    group = [args.player] + partners if partners else DEFAULT_GROUP
    for label, path in [
        ("similarity chart", charts.similarity_chart(args.player)),
        ("role chart", charts.role_chart(args.player)),
        ("compatibility heatmap", charts.compatibility_heatmap(group)),
        ("squad network", charts.squad_network(DEFAULT_GROUP)),
        ("embedding map", charts.embedding_map()),
    ]:
        print(f"[saved] {label}: {path}" if path else f"[skipped] {label}: player not found")


if __name__ == "__main__":
    main()
