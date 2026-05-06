"""
Water Distribution Pipeline Condition Assessment & Failure Prediction
Zindzi Griffin | AI Research Engineer
----------------------------------------------------------------------
Uses EPA ECHO-inspired synthetic dataset modeled on real-world water
infrastructure attributes (pipe age, material, pressure, break history)
to predict failure likelihood and surface high-risk pipeline segments.

Public data reference: EPA ECHO, EBMUD Water Quality Reports,
ASCE Infrastructure Report Card pipe attribute distributions
"""

# ── IMPORTS ───────────────────────────────────────────────────────────────────

import pandas as pd          # pandas is our data table library — think of it as Python's Excel
import numpy as np           # numpy handles all the math and number arrays under the hood

# These are our three machine learning models — each learns differently, we compare them at the end
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# train_test_split divides our data into training and testing portions
# cross_val_score runs the model multiple times on different slices to check consistency
from sklearn.model_selection import train_test_split, cross_val_score

# LabelEncoder converts text categories (like "Cast Iron") into numbers models can read
# StandardScaler rescales all features to the same range so no single feature dominates
from sklearn.preprocessing import LabelEncoder, StandardScaler

# These are our evaluation tools — they measure how well the model performs
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve)

import matplotlib.pyplot as plt      # matplotlib is our main chart drawing library
import matplotlib.patches as mpatches  # patches lets us create custom legend icons
import seaborn as sns                # seaborn makes prettier charts, especially heatmaps
import warnings
warnings.filterwarnings("ignore")    # suppress non-critical warnings to keep output clean

# Setting a random seed means our "random" numbers are the same every run
# This makes the project reproducible — anyone who runs it gets identical results
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: BUILD THE DATASET
# We are creating a synthetic (fake but realistic) dataset of 2,000 pipe segments
# modeled on real-world attribute distributions from EPA ECHO and ASCE studies
# ══════════════════════════════════════════════════════════════════════════════

N = 2000  # total number of pipe segments we are simulating

# Randomly assign a material to each pipe segment
# The probabilities (p) reflect real US water infrastructure composition:
# Cast Iron is most common because it was the standard material for 100+ years
# PVC is newer and more durable, making up a smaller share of aging systems
pipe_materials = np.random.choice(
    ["Cast Iron", "Ductile Iron", "PVC", "Asbestos Cement", "Steel"],
    size=N, p=[0.30, 0.25, 0.20, 0.15, 0.10]
)

# Generate a realistic installation year for each pipe using a normal distribution
# Mean of 1975 means most pipes were installed around that era, with variation of 20 years
# .clip(1920, 2020) ensures no pipe is older than 1920 or newer than 2020
install_year = np.random.normal(1975, 20, N).clip(1920, 2020).astype(int)

# Calculate how old each pipe is by subtracting install year from current year
# Older pipes carry more risk — this becomes one of our most important features
pipe_age = 2024 - install_year

# Assign a diameter to each pipe — diameter affects pressure and failure risk
# Smaller pipes (4-6 inch) are most common in distribution networks
diameter_inches = np.random.choice([4, 6, 8, 12, 16, 24], size=N,
                                    p=[0.15, 0.30, 0.25, 0.20, 0.07, 0.03])

# Operating pressure in PSI (pounds per square inch)
# Normal distribution centered at 65 PSI, clipped between 20 and 120
# Higher sustained pressure accelerates wear especially in older materials
operating_pressure_psi = np.random.normal(65, 15, N).clip(20, 120)

# Soil corrosivity — the aggressiveness of the soil surrounding each pipe
# High corrosivity soil eats through metal pipes much faster
soil_corrosivity = np.random.choice(
    ["Low", "Moderate", "High"], size=N, p=[0.40, 0.35, 0.25]
)

# Number of times each pipe has broken before — a strong predictor of future breaks
# Poisson distribution is appropriate here because breaks are rare discrete events
prior_breaks = np.random.poisson(1.2, N)

# Number of leak complaints logged from residents near each pipe segment
leak_complaints = np.random.poisson(0.8, N)

# How many years ago the pipe was last physically inspected
# A pipe not inspected in 10+ years carries more unknown risk
last_inspection_years = np.random.uniform(0, 15, N)

# Tuberculation index measures internal rust and mineral buildup on pipe walls (0 to 1 scale)
# Beta distribution naturally produces values between 0 and 1, skewed toward lower values
tuberculation_index = np.random.beta(2, 5, N)


# ── CALCULATE FAILURE PROBABILITY ─────────────────────────────────────────────
# This is the ground truth we are trying to predict
# Each factor contributes a weighted amount to the overall failure probability
# These weights are grounded in ASCE and EPA infrastructure failure literature

# Dictionary mapping each material to its baseline failure risk
# Cast Iron and Asbestos Cement are oldest and most fragile — highest risk
# PVC is modern and corrosion-resistant — lowest risk
mat_risk = {"Cast Iron": 0.45, "Asbestos Cement": 0.40, "Steel": 0.25,
            "Ductile Iron": 0.15, "PVC": 0.08}

# Dictionary mapping soil corrosivity to its added risk contribution
soil_risk = {"High": 0.20, "Moderate": 0.10, "Low": 0.02}

# Build the failure probability for every pipe by combining all risk factors:
# - 0.003 * pipe_age:         each year of age adds a small but compounding risk
# - mat_risk lookup:          material class contributes a fixed baseline risk
# - soil_risk lookup:         soil aggressiveness adds environmental risk
# - 0.05 * prior_breaks:      each past break meaningfully raises future risk
# - 0.003 * pressure:         sustained high pressure accelerates wear
# - 0.08 * tuberculation:     internal buildup restricts flow and weakens walls
# - random noise:             real-world data is never perfectly clean
failure_prob = (
    0.003 * pipe_age
    + np.array([mat_risk[m] for m in pipe_materials])
    + np.array([soil_risk[s] for s in soil_corrosivity])
    + 0.05 * prior_breaks
    + 0.003 * operating_pressure_psi
    + 0.08 * tuberculation_index
    + np.random.normal(0, 0.05, N)
).clip(0, 1)   # clip keeps all probabilities between 0 and 1

# Convert continuous probability into a binary label (0 = low risk, 1 = high risk)
# Threshold of 0.55 means a pipe needs multiple risk factors stacking up to be flagged
failure_label = (failure_prob > 0.55).astype(int)

# Assemble everything into a pandas DataFrame — our working data table
df = pd.DataFrame({
    "pipe_id":                [f"EBMUD-{i:04d}" for i in range(N)],  # unique ID for each segment
    "install_year":           install_year,
    "pipe_age_years":         pipe_age,
    "material":               pipe_materials,
    "diameter_inches":        diameter_inches,
    "operating_pressure_psi": operating_pressure_psi,
    "soil_corrosivity":       soil_corrosivity,
    "prior_breaks":           prior_breaks,
    "leak_complaints":        leak_complaints,
    "last_inspection_years":  last_inspection_years,
    "tuberculation_index":    tuberculation_index,
    "failure_probability":    failure_prob,
    "failure_label":          failure_label
})

# Print a quick sanity check on our dataset
print(f"Dataset: {N} pipe segments | Failure rate: {failure_label.mean():.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: FEATURE ENGINEERING
# Raw data rarely gives models the best signal
# We transform and combine existing features to create stronger predictors
# ══════════════════════════════════════════════════════════════════════════════

# Machine learning models require numbers — they cannot read text like "Cast Iron"
# LabelEncoder converts each unique text category into a unique integer
# e.g. "Asbestos Cement"=0, "Cast Iron"=1, "Ductile Iron"=2, "PVC"=3, "Steel"=4
le_mat  = LabelEncoder()
le_soil = LabelEncoder()
df["material_enc"]         = le_mat.fit_transform(df["material"])
df["soil_corrosivity_enc"] = le_soil.fit_transform(df["soil_corrosivity"])

# Interaction feature: age × pressure
# A 90-year-old pipe at 90 PSI is far more dangerous than either factor alone suggests
# Multiplying them together captures this compounding relationship
df["age_pressure_interaction"] = df["pipe_age_years"] * df["operating_pressure_psi"]

# Break rate normalized by age — a pipe with 3 breaks in 10 years is much worse
# than one with 3 breaks in 60 years, but raw break count would treat them the same
df["break_rate_per_decade"]    = df["prior_breaks"] / (df["pipe_age_years"] / 10 + 1)

# Inspection lag weighted by risk — a high-risk pipe not inspected recently
# is more dangerous than a low-risk pipe with the same inspection gap
df["inspection_lag_risk"]      = df["last_inspection_years"] * df["failure_probability"]

# Define our final list of features — these are the inputs the model will learn from
features = [
    "pipe_age_years",           # raw age
    "diameter_inches",          # pipe size
    "operating_pressure_psi",   # pressure load
    "prior_breaks",             # break history
    "leak_complaints",          # community-reported signals
    "last_inspection_years",    # time since last inspection
    "tuberculation_index",      # internal condition
    "material_enc",             # encoded material type
    "soil_corrosivity_enc",     # encoded soil aggressiveness
    "age_pressure_interaction", # engineered: compounding age + pressure risk
    "break_rate_per_decade",    # engineered: normalized break frequency
    "inspection_lag_risk"       # engineered: weighted inspection gap
]

# X is our feature matrix — the inputs the model learns from
# y is our target vector — the labels the model is trying to predict
X = df[features]
y = df["failure_label"]

# StandardScaler rescales every feature to have mean=0 and standard deviation=1
# Without this, features with large numbers (like pressure: 20-120)
# would drown out features with small numbers (like tuberculation: 0-1)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)  # fit learns the scale, transform applies it

# Split data: 80% for training (model learns from this), 20% for testing (we evaluate on this)
# stratify=y ensures both splits have the same ratio of high/low risk segments
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MODEL TRAINING
# We train three different model types and compare their performance
# This is standard practice — no single algorithm is always best
# ══════════════════════════════════════════════════════════════════════════════

# Define our three model candidates with their hyperparameters
# Random Forest: builds many independent decision trees and averages their votes
# Gradient Boosting: builds trees sequentially, each correcting the previous one's errors
# Logistic Regression: fits a mathematical curve to find the decision boundary
models = {
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = {}  # empty dictionary to store each model's performance metrics

# Loop through each model, train it, and evaluate it
for name, model in models.items():

    # .fit() is where the actual learning happens — the model studies the training data
    model.fit(X_train, y_train)

    # Cross-validation: splits the full dataset into 5 folds, trains on 4, tests on 1
    # Repeats 5 times rotating which fold is the test set — gives a reliable average score
    # roc_auc is our scoring metric (explained below)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="roc_auc")

    # Generate predictions on the held-out test set the model has never seen
    y_pred  = model.predict(X_test)           # hard predictions: 0 or 1
    y_proba = model.predict_proba(X_test)[:, 1]  # soft probabilities: 0.0 to 1.0

    # Store all metrics for this model so we can compare later
    results[name] = {
        "model":    model,
        "cv_auc":   cv_scores.mean(),   # average AUC across 5 folds
        "cv_std":   cv_scores.std(),    # standard deviation — lower means more consistent
        "y_pred":   y_pred,
        "y_proba":  y_proba,
        "test_auc": roc_auc_score(y_test, y_proba)  # AUC on the final held-out test set
    }

    # Print a summary for each model
    # AUC (Area Under the ROC Curve): ranges from 0.5 (random) to 1.0 (perfect)
    # It measures how well the model separates high-risk from low-risk segments
    print(f"\n{name}")
    print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Test AUC: {roc_auc_score(y_test, y_proba):.3f}")

    # Classification report shows precision, recall, and F1 for each class
    # Precision: of pipes flagged as high risk, how many actually were?
    # Recall: of all truly high-risk pipes, how many did we catch?
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

# Select the best model based on test AUC score
best_name = max(results, key=lambda k: results[k]["test_auc"])
best      = results[best_name]
print(f"\n✓ Best model: {best_name} (AUC = {best['test_auc']:.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: VISUALIZATIONS
# We generate a 9-panel figure saved as pipeline_analysis.png
# Every chart is produced directly from the model outputs — nothing is hand-drawn
# ══════════════════════════════════════════════════════════════════════════════

# Define a color palette inspired by EBMUD's branding
EBMUD_BLUE  = "#005B8E"   # primary blue — used for standard/acceptable risk
EBMUD_TEAL  = "#00A499"   # teal accent — used for secondary model lines
EBMUD_LIGHT = "#E8F4F8"   # light blue — used for chart backgrounds
ALERT_RED   = "#C0392B"   # red — used to highlight critical/high risk
WARN_AMBER  = "#E67E22"   # amber — used to highlight elevated/moderate risk

# Create a large figure with 3 rows and 3 columns = 9 subplots total
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("white")  # white background for the whole figure


# ── CHART 1: ROC Curves ───────────────────────────────────────────────────────
# ROC curve plots True Positive Rate vs False Positive Rate at every threshold
# A model hugging the top-left corner is near-perfect
# The diagonal dashed line represents a random coin-flip baseline
ax1 = fig.add_subplot(3, 3, 1)   # position 1 in a 3x3 grid
colors = [EBMUD_BLUE, EBMUD_TEAL, WARN_AMBER]  # one color per model
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])  # compute the curve points
    ax1.plot(fpr, tpr, color=color, lw=2,
             label=f"{name} (AUC={res['test_auc']:.3f})")
ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.5)   # diagonal random baseline
ax1.set_xlabel("False Positive Rate", fontsize=9)
ax1.set_ylabel("True Positive Rate", fontsize=9)
ax1.set_title("ROC Curves — Model Comparison", fontweight="bold", fontsize=10)
ax1.legend(fontsize=7)
ax1.set_facecolor(EBMUD_LIGHT)
ax1.grid(alpha=0.3)


# ── CHART 2: Feature Importance ───────────────────────────────────────────────
# Random Forest tracks how much each feature reduces uncertainty across all trees
# Higher importance = that feature is doing more work to separate high/low risk
# Red bars = most impactful features (above 0.12 threshold)
ax2 = fig.add_subplot(3, 3, 2)
rf_model    = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=features).sort_values()

# Human-readable labels to replace the code variable names on the chart
feat_labels = {
    "pipe_age_years":          "Pipe Age (years)",
    "diameter_inches":         "Diameter (in)",
    "operating_pressure_psi":  "Operating Pressure (psi)",
    "prior_breaks":            "Prior Breaks",
    "leak_complaints":         "Leak Complaints",
    "last_inspection_years":   "Inspection Lag (yrs)",
    "tuberculation_index":     "Tuberculation Index",
    "material_enc":            "Pipe Material",
    "soil_corrosivity_enc":    "Soil Corrosivity",
    "age_pressure_interaction":"Age × Pressure",
    "break_rate_per_decade":   "Break Rate/Decade",
    "inspection_lag_risk":     "Inspection Lag Risk"
}
importances.index = [feat_labels[i] for i in importances.index]
colors_fi = [ALERT_RED if v > 0.12 else EBMUD_BLUE for v in importances.values]
importances.plot(kind="barh", ax=ax2, color=colors_fi)
ax2.set_title("Feature Importance (Random Forest)", fontweight="bold", fontsize=10)
ax2.set_xlabel("Importance Score", fontsize=9)
ax2.set_facecolor(EBMUD_LIGHT)
ax2.grid(alpha=0.3, axis="x")


# ── CHART 3: Failure Rate by Material ────────────────────────────────────────
# Simple group-by showing what fraction of each material type was labeled high-risk
# Confirms that Cast Iron and Asbestos Cement are highest risk — matches field literature
ax3 = fig.add_subplot(3, 3, 3)
mat_failure = df.groupby("material")["failure_label"].mean().sort_values(ascending=False)
bar_colors  = [ALERT_RED if v > 0.35 else WARN_AMBER if v > 0.20 else EBMUD_BLUE
               for v in mat_failure.values]
mat_failure.plot(kind="bar", ax=ax3, color=bar_colors, edgecolor="white")
ax3.set_title("Failure Rate by Pipe Material", fontweight="bold", fontsize=10)
ax3.set_ylabel("Failure Rate", fontsize=9)
ax3.set_xticklabels(mat_failure.index, rotation=30, ha="right", fontsize=8)
ax3.set_facecolor(EBMUD_LIGHT)
ax3.grid(alpha=0.3, axis="y")
# Horizontal dashed line showing the dataset-wide average failure rate for reference
ax3.axhline(failure_label.mean(), color="black", ls="--", lw=1.2,
            label=f"Overall avg: {failure_label.mean():.1%}")
ax3.legend(fontsize=8)


# ── CHART 4: Failure Probability Distribution ─────────────────────────────────
# Histogram showing how failure probabilities spread across low vs high risk segments
# Good separation between the two distributions = model has strong signal to work with
ax4 = fig.add_subplot(3, 3, 4)
ax4.hist(df[df.failure_label==0]["failure_probability"], bins=40, alpha=0.7,
         color=EBMUD_BLUE, label="Low Risk Segments")
ax4.hist(df[df.failure_label==1]["failure_probability"], bins=40, alpha=0.7,
         color=ALERT_RED,  label="High Risk Segments")
ax4.set_title("Failure Probability Distribution", fontweight="bold", fontsize=10)
ax4.set_xlabel("Predicted Failure Probability", fontsize=9)
ax4.set_ylabel("Count", fontsize=9)
ax4.legend(fontsize=8)
ax4.set_facecolor(EBMUD_LIGHT)
ax4.grid(alpha=0.3)


# ── CHART 5: Pipe Age vs Failure Probability ──────────────────────────────────
# Scatter plot showing how age and material interact to drive risk
# Each dot is one pipe segment — color coded by material
# You should see older Cast Iron pipes clustering at higher risk scores
ax5 = fig.add_subplot(3, 3, 5)
for mat, color in zip(["Cast Iron", "PVC", "Ductile Iron"],
                      [ALERT_RED, EBMUD_TEAL, EBMUD_BLUE]):
    sub = df[df.material == mat]   # filter to just pipes of this material
    ax5.scatter(sub.pipe_age_years, sub.failure_probability,
                alpha=0.3, s=12, color=color, label=mat)
ax5.set_title("Pipe Age vs. Failure Probability by Material",
              fontweight="bold", fontsize=10)
ax5.set_xlabel("Pipe Age (years)", fontsize=9)
ax5.set_ylabel("Failure Probability", fontsize=9)
ax5.legend(fontsize=8)
ax5.set_facecolor(EBMUD_LIGHT)
ax5.grid(alpha=0.3)


# ── CHART 6: Confusion Matrix ─────────────────────────────────────────────────
# A 2x2 grid showing correct and incorrect predictions for the best model
# Top-left: correctly predicted low risk (true negatives)
# Bottom-right: correctly predicted high risk (true positives)
# Off-diagonal cells are errors — we want these as small as possible
ax6 = fig.add_subplot(3, 3, 6)
cm = confusion_matrix(y_test, best["y_pred"])  # compute the 2x2 matrix
sns.heatmap(cm, annot=True, fmt="d", ax=ax6,   # annot=True prints numbers in cells
            cmap=sns.light_palette(EBMUD_BLUE, as_cmap=True),
            xticklabels=["Low Risk", "High Risk"],
            yticklabels=["Low Risk", "High Risk"])
ax6.set_title(f"Confusion Matrix — {best_name}", fontweight="bold", fontsize=10)
ax6.set_ylabel("Actual", fontsize=9)
ax6.set_xlabel("Predicted", fontsize=9)


# ── CHART 7: District-Level Risk Map ──────────────────────────────────────────
# Aggregates predicted risk scores by service district
# This is the operational output — tells a utility which geographic areas need attention first
ax7 = fig.add_subplot(3, 3, 7)

# Get predicted risk scores for every pipe in the full dataset (not just test set)
rf_proba_full        = results["Random Forest"]["model"].predict_proba(X_scaled)[:, 1]
df["predicted_risk"] = rf_proba_full  # attach scores back to the main dataframe

# Randomly assign each pipe to a service district (in real use, this would come from GIS data)
districts      = [f"District {i}" for i in range(1, 9)]
df["district"] = np.random.choice(districts, N)

# Calculate average risk score per district and sort highest to lowest
dist_risk = df.groupby("district")["predicted_risk"].mean().sort_values(ascending=False)

# Color-code bars by risk tier: red = critical, amber = elevated, blue = acceptable
bar_c2 = [ALERT_RED if v > 0.45 else WARN_AMBER if v > 0.35 else EBMUD_BLUE
          for v in dist_risk.values]
dist_risk.plot(kind="bar", ax=ax7, color=bar_c2, edgecolor="white")
ax7.set_title("Avg. Predicted Risk by Service District",
              fontweight="bold", fontsize=10)
ax7.set_ylabel("Mean Failure Risk Score", fontsize=9)
ax7.set_xticklabels(dist_risk.index, rotation=30, ha="right", fontsize=8)
ax7.set_facecolor(EBMUD_LIGHT)
ax7.grid(alpha=0.3, axis="y")

# Custom legend patches explaining the color tiers
red_p   = mpatches.Patch(color=ALERT_RED,  label="Critical (>0.45)")
amber_p = mpatches.Patch(color=WARN_AMBER, label="Elevated (>0.35)")
blue_p  = mpatches.Patch(color=EBMUD_BLUE, label="Acceptable")
ax7.legend(handles=[red_p, amber_p, blue_p], fontsize=7)


# ── CHART 8: Precision-Recall Curves ──────────────────────────────────────────
# Precision-Recall is especially useful when classes are imbalanced
# Precision: how many flagged pipes are truly high risk?
# Recall: how many truly high-risk pipes did we catch?
# The curve shows the tradeoff — you can increase recall but precision drops, and vice versa
ax8 = fig.add_subplot(3, 3, 8)
for (name, res), color in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
    ax8.plot(rec, prec, color=color, lw=2, label=name)
ax8.set_title("Precision-Recall Curves", fontweight="bold", fontsize=10)
ax8.set_xlabel("Recall", fontsize=9)
ax8.set_ylabel("Precision", fontsize=9)
ax8.legend(fontsize=7)
ax8.set_facecolor(EBMUD_LIGHT)
ax8.grid(alpha=0.3)


# ── CHART 9: Model Performance Summary Table ──────────────────────────────────
# A clean comparison table showing CV AUC, standard deviation, and test AUC
# for all three models side by side — makes the model selection decision transparent
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis("off")  # turn off axis lines — this subplot is just a table, not a chart

# Build the table data as a list of rows
perf_data = [[name, f"{res['cv_auc']:.3f}", f"±{res['cv_std']:.3f}", f"{res['test_auc']:.3f}"]
             for name, res in results.items()]

# Render the table inside this subplot
table = ax9.table(
    cellText=perf_data,
    colLabels=["Model", "CV AUC", "Std Dev", "Test AUC"],
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.0)   # make the table wider and taller for readability

# Style the header row with EBMUD blue background and white text
# Alternate row shading for easier reading
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor(EBMUD_BLUE)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor(EBMUD_LIGHT if r % 2 == 0 else "white")
ax9.set_title("Model Performance Summary", fontweight="bold", fontsize=10, pad=80)


# ── FINALIZE AND SAVE ─────────────────────────────────────────────────────────
# Add a master title across the top of the entire 9-panel figure
plt.suptitle(
    "EBMUD Water Distribution Pipeline Condition Assessment\n"
    "Machine Learning-Based Failure Prediction Framework",
    fontsize=14, fontweight="bold", color=EBMUD_BLUE, y=1.01
)
plt.tight_layout()  # automatically adjust spacing so charts don't overlap

# Save the figure as a PNG file in the current directory
# dpi=150 gives high enough resolution for a README or presentation
# bbox_inches="tight" ensures nothing gets cropped
plt.savefig("pipeline_analysis.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()   # close the figure to free memory
print("\n✓ Visualization saved: pipeline_analysis.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: PRINT TOP 10 HIGHEST-RISK SEGMENTS
# This is the operational output — the ranked list a field engineer would act on
# ══════════════════════════════════════════════════════════════════════════════

# Sort all pipe segments by predicted risk score, take the top 10
# Select only the most relevant columns for a field engineer to read
top10 = (df.nlargest(10, "predicted_risk")
           [["pipe_id", "pipe_age_years", "material", "operating_pressure_psi",
             "prior_breaks", "predicted_risk"]]
           .rename(columns={
               "pipe_id":               "Pipe ID",
               "pipe_age_years":        "Age (yrs)",
               "material":              "Material",
               "operating_pressure_psi":"Pressure (psi)",
               "prior_breaks":          "Prior Breaks",
               "predicted_risk":        "Risk Score"
           }))
top10["Risk Score"] = top10["Risk Score"].round(3)  # round to 3 decimal places for readability
print("\n── Top 10 Highest-Risk Pipeline Segments ──")
print(top10.to_string(index=False))  # to_string prints the full table without truncation

# Final project summary printed to the console
print(f"""
── Project Summary ──────────────────────────────────────────────────
Dataset:        {N} pipe segments | {len(features)} engineered features
Best Model:     {best_name} | Test AUC = {best['test_auc']:.3f}
Use Case:       Prioritize inspection & replacement of high-risk mains
Data Sources:   EPA ECHO attribute distributions, ASCE failure studies,
                EBMUD publicly available water quality reports
GitHub:         github.com/zindzigriffin/water-pipeline-risk-model
─────────────────────────────────────────────────────────────────────
""")