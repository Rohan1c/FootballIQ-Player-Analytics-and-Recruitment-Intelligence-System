"""One-page intelligence report that pulls every engine together for one player."""
from __future__ import annotations

from .compatibility import best_partners
from .scouting import global_lookalikes
from .similarity import similar_players
from .squad import XI_ORDER, best_trio, build_xi
from .store import PlayerStore, get_store
from .transfer import find_replacements

ATTRIBUTE_COLUMNS = [
    "Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physicality", "Vision",
    "Crossing", "Finishing", "Ball Control", "Interceptions", "Standing Tackle",
    "Strength", "Aggression",
]
USAGE = {
    "Creative Playmaker": "Use as primary creator.",
    "Deep Playmaker": "Use as tempo controller.",
    "Ball Winner": "Use as defensive shield.",
    "Box-to-Box": "Use as transition midfielder.",
    "Wide Winger": "Use to stretch play on the flank.",
    "Creative Winger": "Use as a wide playmaker.",
    "Inside Forward": "Use as goalscoring wide attacker.",
    "Poacher": "Use as penalty-box finisher.",
    "Target Forward": "Use as focal point striker.",
    "False 9": "Use as a dropping forward who links play.",
    "Ball Playing Defender": "Use to progress possession from defence.",
    "Defensive Defender": "Use as defensive stopper.",
    "Goalkeeper": "Use in goal.",
}


def top_attributes(store: PlayerStore, i: int, top_n: int = 5) -> list[tuple[str, float]]:
    values = store.players.loc[i, ATTRIBUTE_COLUMNS].astype(float)
    return list(values.sort_values(ascending=False).head(top_n).items())


def _rule(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def generate_report(name: str, store: PlayerStore | None = None) -> bool:
    """Print the report; returns False if the player was not found."""
    store = store or get_store()
    i = store.find(name)
    if i is None:
        print("Player not found.")
        return False
    row, role = store.row(i), store.roles[i]

    _rule("FOOTBALL INTELLIGENCE REPORT")
    print("\nPLAYER PROFILE\n")
    print(f"Name: {row['Player']}")
    print(f"Club: {row['Squads']}")
    print(f"Position: {row['Position']}")
    print(f"Age: {int(row['Age'])}")
    print(f"Minutes: {int(row['Min'])}" + ("  (small sample: treat with care)" if row["Min"] < 450 else ""))
    print(f"Primary Role: {role}")

    print("\nROLE PROBABILITIES\n")
    for role_name, score in sorted(store.role_scores(i).items(), key=lambda x: -x[1])[:5]:
        print(f"{role_name:<25}{score:.3f}")

    if not store.is_gk[i]:
        print("\nTOP ATTRIBUTES\n")
        for attr, value in top_attributes(store, i):
            print(f"{attr:<25}{value:.0f}")

    _rule("MOST SIMILAR PLAYERS")
    for rank, r in enumerate(similar_players(name, 5, store=store), 1):
        shared = ", ".join(n for n, _, _ in r["why"]["shared"]) or "-"
        print(f"{rank}. {r['player']} | {r['position']} | {r['role']} | {r['similarity']:.3f} | shared: {shared}")

    if store.is_gk[i]:
        _rule("END OF REPORT")
        return True

    _rule("BEST COMPATIBLE PARTNERS")
    for rank, r in enumerate(best_partners(name, 5, store=store), 1):
        print(f"{rank}. {r['player']} | {r['position']} | {r['role']} | {r['score']:.3f}")

    _rule("BEST REPLACEMENTS")
    for rank, r in enumerate(find_replacements(name, 5, store=store), 1):
        print(f"{rank}. {r['player']} | {r['position']} | {r['role']} | Age {r['age']} | {r['score']:.3f}")

    _rule("LOOK-ALIKES OUTSIDE THE BIG FIVE LEAGUES")
    for rank, r in enumerate(global_lookalikes(name, 5, store=store), 1):
        print(f"{rank}. {r['player']} | {r['position']} | Age {r['age']} | {r['team']} | {r['similarity']:.3f}")

    _rule("BEST MIDFIELD TRIO")
    trio, trio_score = best_trio(i, store=store)
    if trio is None:
        print("Not applicable (the player is not a midfielder).")
    else:
        for j in trio:
            print(f"{store.names[j]} | {store.positions[j]} | {store.roles[j]}")
        print(f"\nTactical Score: {trio_score:.3f}")

    _rule("RECOMMENDED STARTING XI")
    xi = build_xi(i, store=store)
    for slot in XI_ORDER:
        j = xi[slot]
        print(f"{slot:<5}{store.names[j]:<30}{store.positions[j]:<6}{store.roles[j]}")

    _rule("FOOTBALL INTELLIGENCE SUMMARY")
    print(f"\n{row['Player']} is primarily classified as a {role} operating from {row['Position']}.")
    print("\nKey Strengths:")
    for attr, _ in top_attributes(store, i, 3):
        print(f"- {attr}")
    print("\nRecommended Usage:")
    print(f"- {USAGE.get(role, 'No specific recommendation.')}")
    _rule("END OF REPORT")
    return True


if __name__ == "__main__":
    generate_report(input("Enter player name: "))
