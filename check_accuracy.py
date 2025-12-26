import json
import os

results_dir = "experiments/results"
results = []

for d in os.listdir(results_dir):
    pred_file = os.path.join(results_dir, d, "test_predictions.json")
    if os.path.exists(pred_file):
        with open(pred_file) as f:
            preds = json.load(f)
            correct = sum(1 for p in preds if p.get("correct", False))
            accuracy = correct / len(preds) if preds else 0
            results.append({"experiment": d, "accuracy": accuracy})

print("=" * 50)
print("ACTUAL ACCURACY FROM PREDICTIONS")
print("=" * 50)
for r in sorted(results, key=lambda x: x["accuracy"], reverse=True)[:15]:
    print(f"{r['experiment']}: {r['accuracy']*100:.1f}%")
