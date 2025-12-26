import os
import json

results = []
results_dir = "experiments/results_gpu"

for exp_dir in sorted(os.listdir(results_dir)):
    results_file = os.path.join(results_dir, exp_dir, "final_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
            results.append({
                "name": exp_dir,
                "accuracy": data.get("accuracy", 0),
                "method": data.get("method", ""),
                "lr": data.get("learning_rate", ""),
            })

# Sort by accuracy
results.sort(key=lambda x: x["accuracy"], reverse=True)

print("=" * 60)
print("EXPERIMENT RESULTS SUMMARY (29 experiments)")
print("=" * 60)
print(f"{'Experiment':<35} | {'Acc':>8} | {'Method':>10}")
print("-" * 60)
for r in results:
    print(f"{r['name']:<35} | {r['accuracy']:>7.1%} | {r['method']:>10}")

print("\n" + "=" * 60)
print("TOP 5 BEST PERFORMING:")
print("-" * 60)
for r in results[:5]:
    print(f"  {r['name']}: {r['accuracy']:.1%}")
print("=" * 60)
