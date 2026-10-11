import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import numpy as np
import matplotlib.pyplot as plt

from legacy.models.prototype_role_engine import (
    find_player,
    get_role_scores
)

player_name = input(
    "Enter player name: "
)

player = find_player(
    player_name
)

if player is None:

    print(
        "Player not found."
    )

    exit()

scores = get_role_scores(
    player
)

sorted_scores = sorted(

    scores.items(),

    key=lambda x: x[1],

    reverse=True

)

top_roles = sorted_scores[:5]

roles = [
    role
    for role, score
    in top_roles
]

values = [
    score
    for role, score
    in top_roles
]

plt.style.use(
    "ggplot"
)

fig, ax = plt.subplots(
    figsize=(12,7)
)

colors = plt.cm.plasma(
    np.linspace(
        0.2,
        0.9,
        len(values)
    )
)

bars = ax.barh(

    roles,

    values,

    color=colors,

    edgecolor="black",

    linewidth=1.2

)

ax.invert_yaxis()

for bar in bars:

    width = bar.get_width()

    ax.text(

        width + 0.005,

        bar.get_y()
        + bar.get_height()/2,

        f"{width:.3f}",

        va="center",

        fontsize=11,

        fontweight="bold"

    )

primary_role = top_roles[0][0]
primary_score = top_roles[0][1]

ax.set_title(

    f"FootballIQ Role Analysis\n{player['Player']}",

    fontsize=20,

    fontweight="bold",

    pad=20

)

ax.set_xlabel(

    "Role Similarity Score",

    fontsize=12,

    fontweight="bold"

)

ax.set_ylabel(

    "FootballIQ Archetypes",

    fontsize=12,

    fontweight="bold"

)

ax.grid(

    axis="x",

    linestyle="--",

    alpha=0.4

)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.figtext(

    0.5,

    0.02,

    f"Primary Role: {primary_role} ({primary_score:.3f})",

    ha="center",

    fontsize=12,

    fontweight="bold"

)

plt.tight_layout()

plt.show()