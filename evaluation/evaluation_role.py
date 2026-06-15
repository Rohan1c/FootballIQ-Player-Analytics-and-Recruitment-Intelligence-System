import os
import sys
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from models.prototype_role_engine import (
    find_player,
    get_primary_role,
    get_role_scores
)

labels = pd.read_csv(
    "evaluation/role_validation_labels.csv",
    header=None,
    names=["player_name", "ground_truth_role"],
    skipinitialspace=True
)

y_true = []
y_pred = []

correct = 0
top2_correct = 0
total = 0

print("\n" + "=" * 100)
print("FOOTBALLIQ ROLE ENGINE EVALUATION")
print("=" * 100 + "\n")

for _, row in labels.iterrows():

    player_name = row["player_name"]
    actual = row["ground_truth_role"]

    player = find_player(player_name)

    if player is None:
        print(f"NOT FOUND: {player_name}")
        continue

    predicted = get_primary_role(player)

    scores = get_role_scores(player)

    sorted_roles = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top2 = [role for role, score in sorted_roles[:2]]

    print(f"\n{player_name}")
    print(f"Actual Role : {actual}")
    print(f"Predicted   : {predicted}")
    print("Top Roles   :")

    for role, score in sorted_roles[:3]:
        print(f"   {role:<25} {score:.4f}")

    if predicted == actual:
        correct += 1

    if actual in top2:
        top2_correct += 1

    y_true.append(actual)
    y_pred.append(predicted)

    total += 1

accuracy = correct / total
top2_accuracy = top2_correct / total

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

print(f"\nTotal Players Evaluated : {total}")
print(f"Top-1 Accuracy          : {accuracy:.3f}")
print(f"Top-2 Accuracy          : {top2_accuracy:.3f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

os.makedirs("evaluation/results", exist_ok=True)

labels_sorted = sorted(
    list(set(y_true) | set(y_pred))
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels_sorted
)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels_sorted
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45,
    colorbar=False
)

plt.title("FootballIQ Role Archetype Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "evaluation/results/role_confusion_matrix.png",
    dpi=300
)

plt.close()

metrics_df = pd.DataFrame({
    "Metric": [
        "Top-1 Accuracy",
        "Top-2 Accuracy"
    ],
    "Value": [
        accuracy,
        top2_accuracy
    ]
})

metrics_df.to_csv(
    "evaluation/results/role_metrics.csv",
    index=False
)

print("\nSaved:")
print("evaluation/results/role_confusion_matrix.png")
print("evaluation/results/role_metrics.csv")