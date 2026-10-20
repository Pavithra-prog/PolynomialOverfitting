
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend to save the screenshot as a PNG.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import LogLocator, LogFormatterMathtext

# ============================================================
# PROJECT 20: POLYNOMIAL OVERFITTING
# Assigned to: PAVITHRA P
# Emp ID: TFTPL435
# Project No.: 20
# ============================================================

# 1. Configuration
SEED = 13
NOISE_LEVEL = 0.25
N_SAMPLES = 60
MAX_DEGREE = 15
OUTPUT_DIR = "."

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# 2. Generate noisy sine-wave data on a jittered grid
x = np.linspace(0, 1, N_SAMPLES)
x += np.random.uniform(-0.007, 0.007, N_SAMPLES)
x = np.clip(x, 0, 1)

y = np.sin(2 * np.pi * x)
y += np.random.normal(0, NOISE_LEVEL, N_SAMPLES)

data = pd.DataFrame({"x": x, "y": y})

# 3. Split into 30 training and 30 validation samples
train = data.sample(n=30, random_state=SEED)
validation = data.drop(train.index)

# 4. Rescale x from [0, 1] to [-1, 1]
def scale_x(values):
    return 2 * np.asarray(values) - 1

x_train = scale_x(train["x"])
y_train = train["y"].to_numpy()

x_val = scale_x(validation["x"])
y_val = validation["y"].to_numpy()

# Sort the points for clean plotting
train = train.sort_values("x")
validation = validation.sort_values("x")

# 5. Fit polynomial models from degree 1 to 15
results = []
models = {}

for degree in range(1, MAX_DEGREE + 1):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefficients = np.polyfit(x_train, y_train, degree)

    models[degree] = coefficients

    train_predictions = np.polyval(coefficients, x_train)
    val_predictions = np.polyval(coefficients, x_val)

    train_rmse = float(
        np.sqrt(np.mean((y_train - train_predictions) ** 2))
    )
    val_rmse = float(
        np.sqrt(np.mean((y_val - val_predictions) ** 2))
    )

    largest_coefficient = float(np.max(np.abs(coefficients)))

    results.append({
        "degree": degree,
        "train_rmse": train_rmse,
        "validation_rmse": val_rmse,
        "largest_coefficient": largest_coefficient,
    })

error_table = pd.DataFrame(results)

# 6. Select the best model using validation error
best_row_index = error_table["validation_rmse"].idxmin()
best_degree = int(error_table.loc[best_row_index, "degree"])
best_val_rmse = float(
    error_table.loc[best_row_index, "validation_rmse"]
)

# Find the simplest degree within 10% of the best validation RMSE
threshold = best_val_rmse * 1.10
eligible = error_table[
    error_table["validation_rmse"] <= threshold
]
simplest_near_best = int(eligible["degree"].min())

degree_1 = error_table.loc[error_table["degree"] == 1].iloc[0]
degree_15 = error_table.loc[error_table["degree"] == 15].iloc[0]

ratio = (
    float(degree_15["validation_rmse"]) / best_val_rmse
    if best_val_rmse > 0 else float("inf")
)

coef_degree_1 = float(degree_1["largest_coefficient"])
coef_degree_15 = float(degree_15["largest_coefficient"])

# 7. Build the multi-panel figure
fig = plt.figure(figsize=(12, 6.3), constrained_layout=False)

gs = fig.add_gridspec(
    2, 6,
    left=0.065,
    right=0.985,
    top=0.92,
    bottom=0.17,
    wspace=0.48,
    hspace=0.48,
)

ax_under = fig.add_subplot(gs[0, 0:2])
ax_good = fig.add_subplot(gs[0, 2:4])
ax_over = fig.add_subplot(gs[0, 4:6])
ax_error = fig.add_subplot(gs[1, 0:4])
ax_coef = fig.add_subplot(gs[1, 4:6])

x_plot = np.linspace(0, 1, 500)
x_plot_scaled = scale_x(x_plot)
y_true = np.sin(2 * np.pi * x_plot)

def plot_fit(ax, degree, title):
    coefficients = models[degree]
    predictions = np.polyval(coefficients, x_plot_scaled)

    row = error_table.loc[
        error_table["degree"] == degree
    ].iloc[0]

    # True sine curve and fitted polynomial
    ax.plot(
        x_plot, y_true,
        color="gray", linestyle="--",
        linewidth=1.1, label="true sine"
    )
    ax.plot(
        x_plot, predictions,
        color="black", linewidth=1.7,
        label=f"degree {degree} fit"
    )

    # Training points and held-out validation points
    ax.scatter(
        train["x"], train["y"],
        s=17, color="#2878E3", edgecolors="white",
        linewidths=0.35, label="train", zorder=3
    )
    ax.scatter(
        validation["x"], validation["y"],
        s=17, facecolors="white", edgecolors="#FF5722",
        linewidths=0.9, label="validation", zorder=3
    )

    ax.set_title(
        f"{title}\n"
        f"train {row['train_rmse']:.2f}, "
        f"validation {row['validation_rmse']:.2f}"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-2, 2)
    ax.legend(
        loc="upper center",
        ncol=2,
        fontsize=6,
        frameon=True,
        borderpad=0.35,
        handlelength=1.5,
        columnspacing=0.8,
    )

plot_fit(ax_under, 1, "Underfit: degree 1")
plot_fit(ax_good, best_degree, f"Good fit: degree {best_degree}")
plot_fit(ax_over, 15, "Overfit: degree 15")

# 8. Training and validation RMSE versus polynomial degree
degrees = error_table["degree"]
train_errors = error_table["train_rmse"]
val_errors = error_table["validation_rmse"]

ax_error.plot(
    degrees, train_errors,
    "-o", color="#2878E3", markersize=4,
    linewidth=1.2, label="train error"
)
ax_error.plot(
    degrees, val_errors,
    "-o", color="#FF5722", markersize=4,
    linewidth=1.2, label="validation error"
)
ax_error.axhline(
    NOISE_LEVEL,
    color="gray", linestyle=":", linewidth=1.2,
    label="noise level"
)
ax_error.axvline(
    best_degree,
    color="black", linestyle="--", linewidth=0.9
)
ax_error.scatter(
    [best_degree], [best_val_rmse],
    color="#2878E3", edgecolors="black",
    s=32, zorder=5
)

ax_error.annotate(
    f"best degree = {best_degree}",
    xy=(best_degree, best_val_rmse),
    xytext=(best_degree + 0.35, best_val_rmse * 2.2),
    arrowprops=dict(arrowstyle="->", linewidth=0.8),
    fontsize=8,
)

ax_error.set_title(
    "Training error keeps falling, validation error turns up"
)
ax_error.set_xlabel("Polynomial degree")
ax_error.set_ylabel("RMSE (log scale)")
ax_error.set_yscale("log")
ax_error.set_xticks(range(1, MAX_DEGREE + 1))
ax_error.set_xlim(0.5, 15.7)
ax_error.legend(loc="upper left", frameon=True)

# 9. Largest absolute polynomial coefficient
ax_coef.plot(
    degrees,
    error_table["largest_coefficient"],
    "-o", color="black", markersize=4, linewidth=1.2
)
ax_coef.set_title("Why: coefficients explode")
ax_coef.set_xlabel("Polynomial degree")
ax_coef.set_ylabel("Largest coefficient (log scale)")
ax_coef.set_yscale("log")
ax_coef.set_xticks(range(2, MAX_DEGREE + 1, 2))
ax_coef.set_xlim(1.5, 15.5)

# Figure caption
fig.text(
    0.5, 0.075,
    "Figure 20. Polynomial fits showing underfit, good fit and overfit, "
    "with training and validation error by degree and coefficient growth.",
    ha="center", va="center",
    fontsize=10, color="#555555", style="italic",
    wrap=True,
)

fig.text(
    0.5, 0.025,
    "Project 20  |  PAVITHRA P  |  Emp ID: TFTPL435",
    ha="center", va="center", fontsize=8, color="#555555"
)

# Save figure at 120 dpi
chart_path = os.path.join(OUTPUT_DIR, "chart.png")
fig.savefig(chart_path, dpi=120, facecolor="white")
plt.close(fig)

# 10. Save the model comparison table
csv_path = os.path.join(OUTPUT_DIR, "error_table.csv")
error_table.to_csv(csv_path, index=False, float_format="%.6f")

# Build summary sentences using f-strings
summary = [
    (
        f"Best degree is {best_degree}: validation RMSE "
        f"{best_val_rmse:.3f} against a noise floor of {NOISE_LEVEL:.2f}."
    ),
    (
        f"Degree 1 underfits: train RMSE "
        f"{degree_1['train_rmse']:.2f}, validation RMSE "
        f"{degree_1['validation_rmse']:.2f}."
    ),
    (
        f"Degree 15 overfits: train RMSE "
        f"{degree_15['train_rmse']:.2f}, validation RMSE "
        f"{degree_15['validation_rmse']:.2f}."
    ),
    (
        f"At degree 15 the validation error is {ratio:.2f} "
        f"times the best model's."
    ),
    (
        f"The simplest degree within 10% of the best validation error "
        f"is {simplest_near_best}; the best is {best_degree}."
    ),
    (
        f"The largest coefficient grew from {coef_degree_1:.3f} "
        f"at degree 1 to {coef_degree_15:.3f} at degree 15."
    ),
]

# Include all metrics and summary statements in results.json
json_data = {
    "project": "Project 20: Polynomial Overfitting",
    "assigned_to": "PAVITHRA P",
    "emp_id": "TFTPL435",
    "project_number": 20,
    "random_seed": SEED,
    "noise_level": NOISE_LEVEL,
    "samples": N_SAMPLES,
    "training_samples": len(train),
    "validation_samples": len(validation),
    "best_degree": best_degree,
    "best_validation_rmse": best_val_rmse,
    "noise_floor": NOISE_LEVEL,
    "simplest_degree_within_10_percent": simplest_near_best,
    "degree_1": {
        "train_rmse": float(degree_1["train_rmse"]),
        "validation_rmse": float(degree_1["validation_rmse"]),
        "largest_coefficient": coef_degree_1,
    },
    "degree_15": {
        "train_rmse": float(degree_15["train_rmse"]),
        "validation_rmse": float(degree_15["validation_rmse"]),
        "largest_coefficient": coef_degree_15,
    },
    "degree_15_validation_error_ratio": ratio,
    "summary": summary,
    "files": {
        "chart": "chart.png",
        "metrics": "error_table.csv",
    },
}

json_path = os.path.join(OUTPUT_DIR, "results.json")
with open(json_path, "w", encoding="utf-8") as file:
    json.dump(json_data, file, indent=4, allow_nan=False)

# Print results to the terminal
print("=" * 65)
print("PROJECT 20: POLYNOMIAL OVERFITTING")
print("Assigned to: PAVITHRA P | Emp ID: TFTPL435")
print("=" * 65)

for sentence in summary:
    print("-", sentence)

print("\nMODEL COMPARISON")
print(error_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nOUTPUT FILES")
print(f"Chart: {os.path.abspath(chart_path)}")
print(f"JSON:  {os.path.abspath(json_path)}")
print(f"CSV:   {os.path.abspath(csv_path)}")
print("\nProject completed successfully.")
