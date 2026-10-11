"""
FootballIQ - main dashboard (terminal menu).

Usage:
    python footballiq_dashboard.py
"""
import os
import sys

if hasattr(sys.stdout, "reconfigure"):          # player names contain accents (Windows consoles)
    sys.stdout.reconfigure(encoding="utf-8")

from footballiq import charts
from footballiq.compatibility import compatibility_score
from footballiq.history import get_history, print_similar_seasons, print_style_timeline
from footballiq.report import generate_report
from footballiq.scouting import print_global_lookalikes
from footballiq.similarity import similar_players
from footballiq.squad import best_trio, build_xi, print_trio, print_xi
from footballiq.store import get_store
from footballiq.transfer import find_replacements


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def header():
    print("=" * 58)
    print("       FootballIQ  --  Football Intelligence Platform")
    print("=" * 58)


def section(title):
    print(f"\n{'─' * 58}\n  {title}\n{'─' * 58}")


def ask(prompt="Enter player name: "):
    return input(f"\n  {prompt}").strip()


def pause():
    input("\n  [Press Enter to return to menu]")


def ask_player(prompt="Enter player name: "):
    """Ask for a name and resolve it; returns (index, matched name) or (None, None)."""
    store = get_store()
    index = store.find(ask(prompt))
    if index is None:
        print("  Player not found.")
        return None, None
    return index, store.names[index]


def offer_chart(question, draw):
    if input(f"\n  {question} (y/n): ").strip().lower() == "y":
        print(f"  [saved] {draw()}")


def menu_similar_players():
    section("1. Similar Players")
    _, name = ask_player()
    if name:
        results = similar_players(name, top_n=10)
        print(f"\n  Players most similar to {name} (same position group):\n")
        for rank, r in enumerate(results, 1):
            shared = ", ".join(n for n, _, _ in r["why"]["shared"]) or "-"
            flag = " *" if r["low_sample"] else ""
            print(f"  {rank:>2}. {r['player']:<26} {r['squad']:<20} {r['similarity']:.3f}  shared: {shared}{flag}")
        if any(r["low_sample"] for r in results):
            print("\n  * fewer than 450 minutes played: statistics are shrunk towards the average")
        offer_chart("Save similarity chart?", lambda: charts.similarity_chart(name))
    pause()


def menu_role_analysis():
    section("2. Role Analysis")
    index, name = ask_player()
    if name:
        store = get_store()
        primary = store.roles[index]
        print(f"\n  Player       : {name}  ({store.positions[index]})")
        print(f"  Primary Role : {primary}\n")
        print(f"  {'Role':<28} {'Prob.':>6}")
        print(f"  {'─' * 28} {'─' * 6}  {'─' * 20}")
        for role, score in sorted(store.role_scores(index).items(), key=lambda x: -x[1])[:5]:
            marker = "  <-- primary" if role == primary else ""
            print(f"  {role:<28} {score:>6.3f}  {'#' * int(score * 30)}{marker}")
        offer_chart("Save role probability chart?", lambda: charts.role_chart(name))
    pause()


def menu_transfer():
    section("3. Transfer Recommendations")
    _, name = ask_player()
    if name:
        younger = input("  Only younger players? (y/n): ").strip().lower() == "y"
        print(f"\n  Replacements for {name}:\n")
        print(f"  {'#':<4} {'Player':<26} {'Club':<20} {'Pos':<5} {'Age':<4} {'Role':<22} {'Score':>6}")
        for rank, r in enumerate(find_replacements(name, top_n=10, younger_only=younger), 1):
            print(f"  {rank:<4} {r['player']:<26} {r['squad']:<20} {r['position']:<5} "
                  f"{r['age']:<4} {r['role']:<22} {r['score']:>6.3f}")
    pause()


def menu_compatibility():
    section("4. Compatibility Analysis")
    print("  Enter up to 5 players (leave blank to stop):\n")
    store, indices = get_store(), []
    for k in range(5):
        text = input(f"  Player {k + 1}: ").strip()
        if not text:
            break
        index = store.find(text)
        if index is None:
            print(f"  [skip] '{text}' not found.")
        else:
            indices.append(index)
    if len(indices) < 2:
        print("  Need at least 2 valid players.")
        pause()
        return
    names = [store.names[i] for i in indices]
    print("\n  Compatibility Matrix:\n")
    print(f"  {'':26}" + "".join(f"{n[:10]:>12}" for n in names))
    for i, name in zip(indices, names):
        print(f"  {name:<26}" + "".join(f"{compatibility_score(i, j):>12.3f}" for j in indices))
    offer_chart("Save compatibility heatmap?", lambda: charts.compatibility_heatmap(names))
    pause()


def menu_squad_builder():
    section("5. Squad Builder  --  Best Midfield Trio")
    index, name = ask_player("Build the midfield around: ")
    if name:
        print_trio(*best_trio(index))
    pause()


def menu_starting_xi():
    section("6. Starting XI Builder  --  4-3-3")
    index, name = ask_player("Build XI around player: ")
    if name:
        print_xi(build_xi(index))
    pause()


def menu_report():
    section("7. Football Intelligence Report")
    _, name = ask_player()
    if name:
        generate_report(name)
    pause()


def menu_global_scouting():
    section("8. Global Scouting  --  look-alikes outside the big five leagues")
    _, name = ask_player()
    if name:
        text = input("  Maximum age (blank for any): ").strip()
        print_global_lookalikes(name, top_n=10, max_age=int(text) if text.isdigit() else None)
    pause()


def menu_history():
    section("9. Historical Explorer  --  2017-18 to 2021-22, with passing and pressing data")
    name = ask()
    history = get_history()
    if history.find(name) is None:
        print("  Player not found in the 2017-22 corpus (needs a season with 900+ minutes).")
        pause()
        return
    text = input("  Season end year, 2018-2022 (blank for his latest): ").strip()
    season = int(text) if text.isdigit() else None
    only = input("  Only players available that same season? (y/n): ").strip().lower() == "y"
    print_similar_seasons(name, season, top_n=10, same_season=only)
    print_style_timeline(name)
    pause()


def menu_save_all():
    section("10. Save All Visualizations")
    player = ask("Focal player for charts (default: Pedri): ") or "Pedri"
    _, name = (None, None) if get_store().find(player) is None else (None, player)
    if name is None:
        print("  Player not found.")
    else:
        group = ["Pedri", "Rodri", "Federico Valverde", "Jude Bellingham", "Cole Palmer", "Moisés Caicedo"]
        for path in (charts.similarity_chart(name), charts.role_chart(name),
                     charts.compatibility_heatmap(group), charts.squad_network(group),
                     charts.embedding_map()):
            print(f"  [saved] {path}")
    pause()


MENU = [
    ("Similar Players", menu_similar_players),
    ("Role Analysis", menu_role_analysis),
    ("Transfer Recommendations", menu_transfer),
    ("Compatibility Analysis", menu_compatibility),
    ("Squad Builder (Midfield Trio)", menu_squad_builder),
    ("Starting XI Builder (4-3-3)", menu_starting_xi),
    ("Football Intelligence Report", menu_report),
    ("Global Scouting (12,000 players)", menu_global_scouting),
    ("Historical Explorer (2017-22, rich data)", menu_history),
    ("Save All Visualizations", menu_save_all),
]


def main():
    print("\n  Loading players...")
    get_store()
    while True:
        clear()
        header()
        print()
        for number, (label, _) in enumerate(MENU, 1):
            print(f"  {number:>2}.  {label}")
        print("\n   0.  Exit\n")
        choice = input("  Select option: ").strip()
        if choice == "0":
            print("\n  Goodbye.\n")
            break
        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            clear()
            header()
            MENU[int(choice) - 1][1]()
        else:
            print("  Invalid option.")
            pause()


if __name__ == "__main__":
    main()
