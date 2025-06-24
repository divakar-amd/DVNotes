import os
import re
import matplotlib.pyplot as plt

folders = {
    "notFix": "bench_results_notFix",
    "pytorchFix": "bench_results_Fix"
}

def extract_metrics(filepath):
    median_ttft = None
    median_tpot = None
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if "Median TTFT" in line:
                median_ttft = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
            if "Median TPOT" in line:
                median_tpot = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
    return median_ttft, median_tpot

def collect_results(folder):
    results = {}
    for fname in os.listdir(folder):
        match = re.match(r"result_(\d+)qps\.txt", fname)
        if match:
            qps = int(match.group(1))
            ttft, tpot = extract_metrics(os.path.join(folder, fname))
            results[qps] = {"Median TTFT": ttft, "Median TPOT": tpot}
    return results

all_results = {name: collect_results(path) for name, path in folders.items()}

qps_values = sorted(set(all_results["notFix"].keys()) | set(all_results["pytorchFix"].keys()))

plt.figure(figsize=(10, 5))

# Plot Median TTFT
plt.subplot(1, 2, 1)
for label, results in all_results.items():
    y = [results.get(qps, {}).get("Median TTFT") for qps in qps_values]
    plt.plot(qps_values, y, marker='o', label=label)
plt.title("Median TTFT vs QPS")
plt.xlabel("QPS")
plt.ylabel("Median TTFT (ms)")
plt.legend()
plt.grid(True)

# Plot Median TPOT
plt.subplot(1, 2, 2)
for label, results in all_results.items():
    y = [results.get(qps, {}).get("Median TPOT") for qps in qps_values]
    plt.plot(qps_values, y, marker='o', label=label)
plt.title("Median TPOT vs QPS")
plt.xlabel("QPS")
plt.ylabel("Median TPOT (ms)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("benchmark_comparison.png")  # Save the figure to a file
# plt.show()  # Commented out to avoid displaying the plot
