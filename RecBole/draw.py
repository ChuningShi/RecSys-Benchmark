import re
import pandas as pd
import matplotlib.pyplot as plt

# Define the log file path
log_file_path = "extracted_results.txt"  # Replace with actual path if needed

# Read the log file
with open(log_file_path, "r") as file:
    lines = file.readlines()

# Initialize lists to store extracted data
epochs = []
valid_scores = []
metrics = {
    "recall@1": [], "recall@5": [], "recall@10": [], "recall@20": [], "recall@50": [], "recall@100": [],
    "ndcg@1": [], "ndcg@5": [], "ndcg@10": [], "ndcg@20": [], "ndcg@50": [], "ndcg@100": [],
    "hit@1": [], "hit@5": [], "hit@10": [], "hit@20": [], "hit@50": [], "hit@100": []
}

# Parsing the log file
current_epoch = None

for line in lines:
    epoch_match = re.search(r"epoch (\d+) evaluating .* valid_score: ([\d.]+)", line)
    if epoch_match:
        current_epoch = int(epoch_match.group(1))
        valid_scores.append(float(epoch_match.group(2)))
        epochs.append(current_epoch)

    metric_match = re.findall(r"(recall|ndcg|hit)@(\d+) : ([\d.]+)", line)
    if metric_match and current_epoch is not None:
        for metric_type, metric_level, value in metric_match:
            metric_key = f"{metric_type}@{metric_level}"
            metrics[metric_key].append(float(value))

# Convert to DataFrame
df_metrics = pd.DataFrame(metrics)
df_metrics["Epoch"] = epochs
df_metrics["Valid Score"] = valid_scores

# Plot trends
plt.figure(figsize=(12, 8))
for column in df_metrics.columns[:-1]:
    plt.plot(df_metrics["Epoch"], df_metrics[column], label=column)

plt.xlabel("Epoch")
plt.ylabel("Metric Value")
plt.title("Trend of Evaluation Metrics Over Epochs")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()
