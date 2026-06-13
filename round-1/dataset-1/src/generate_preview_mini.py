#!/usr/bin/env python3
"""Generate preview and mini versions of the dataset."""

from pathlib import Path
import json

# Load full dataset
with open("full_data_out.json") as f:
    data = json.load(f)

# Generate preview (3 examples per dataset)
preview = {"datasets": []}
for dataset in data["datasets"]:
    preview["datasets"].append({
        "dataset": dataset["dataset"],
        "examples": dataset["examples"][:3]
    })

with open("preview_full_data_out.json", "w") as f:
    json.dump(preview, f, indent=2)

# Generate mini (10 examples per dataset)
mini = {"datasets": []}
for dataset in data["datasets"]:
    mini["datasets"].append({
        "dataset": dataset["dataset"],
        "examples": dataset["examples"][:10]
    })

with open("mini_full_data_out.json", "w") as f:
    json.dump(mini, f, indent=2)

print(f"Generated preview and mini versions")
print(f"Preview: {sum(len(d['examples']) for d in preview['datasets'])} examples")
print(f"Mini: {sum(len(d['examples']) for d in mini['datasets'])} examples")
