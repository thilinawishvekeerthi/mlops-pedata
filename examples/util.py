from pedata.util import (
    OptimizationObjective,
    get_summary_variable,
    append_summary_variable,
    zscore,
)
import numpy as np
import datasets as ds

# ======================================================
print("=== OptimizationObjective===")
# For 'min' direction with default values for aim_for and weight
objective_min = OptimizationObjective(direction="min")
print(f"objective_min: {objective_min}")
# OptimizationObjective(direction='min', aim_for=None, weight=1)

# For 'fix' direction with a specific aim_for value and default weight
objective_fix = OptimizationObjective(direction="fix", aim_for=0.5, weight=5)
print(f"objective_fix: {objective_fix}")
# OptimizationObjective(direction='fix', aim_for=42.0, weight=1)

# For 'max' direction with custom values for aim_for and weight
objective_max = OptimizationObjective(direction="max", weight=2)
print(f"objective_max: {objective_max}")
# OptimizationObjective(direction='max', aim_for=None, weight=2)

# ======================================================
print("============================")
print("=== get_summary_variable ===")
dataset = ds.Dataset.from_dict(
    {"target1": np.arange(10) + 1, "target2": np.arange(10), "target3": np.arange(10)}
)
print("dataset:")
print(f"{dataset.to_pandas().head(10)}")
normalised_summary = get_summary_variable(dataset)
print("normalised_summary:")
print(f"{normalised_summary}")

# ======================================================
print("============================")
print("=== adding target summary variable to a dataset ===")
dataset = dataset.add_column("target summary variable", get_summary_variable(dataset))

# ======================================================
print("============================")
print("=== adding target summary variable to a dataset - with objective ===")
dataset = ds.Dataset.from_dict(
    {
        "target1": np.arange(10, dtype=np.float64),
        "target2": np.random.random(10),
        "target3": np.arange(10, dtype=np.float64)[::-1],
    }
)
print("dataset:")
print(f"{dataset.to_pandas().head(10)}")

objectives = {
    "target1": OptimizationObjective(direction="min", weight=2),
    "target2": OptimizationObjective(direction="max", weight=1),
    "target3": OptimizationObjective(direction="fix", aim_for=0.5, weight=10),
}

dataset = append_summary_variable(
    dataset,
    normalization=zscore,
    objectives=objectives,
    summary_variable_name="target summary variable",
)

print("dataset with summary variables:")
print(f"{dataset.to_pandas().head(10)}")
