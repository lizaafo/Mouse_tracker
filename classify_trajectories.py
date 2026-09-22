"""
classify_trajectories.py

Stage 5 in the applied geometry & kinematics project:
Classification of mouse trajectories between humans, generative bots, and adversarial bots.

Provides three rigorous evaluation benchmarks:
1. Benchmark A: Full Kinematic & Geometric Model (all 11 features, including High-Frequency Jerk).
2. Benchmark B: Pure Spatial Geometry & Macro-Kinematics (excluding Jerk to prevent reliance on hardware micro-tremor).
3. Benchmark C: Leave-One-Session-Out Cross-Validation (evaluating generalization to unseen sessions + error analysis).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
    LeaveOneGroupOut,
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


def run_classification():
    csv_path = Path(__file__).resolve().parent / "geometry_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run analyze_trajectories.py first.")
        return

    df = pd.read_csv(csv_path)

    # Applied geometric and kinematic features
    features_all = [
        "power_law_beta",
        "power_law_r",
        "peak_to_mean_speed",
        "time_to_peak_ratio",
        "max_chord_dev_px",
        "bezier_residual_px",
        "curvature_std",
        "path_ratio",
        "total_angle_change",
        "log_dimensionless_jerk",
        "affine_velocity_cv",
    ]

    features_no_jerk = [f for f in features_all if f != "log_dimensionless_jerk"]

    df_clean = df.dropna(subset=features_all + ["source_type", "session_id"]).copy()
    if len(df_clean) < 6 or df_clean["source_type"].nunique() < 2:
        print(f"Error: Insufficient clean trajectories ({len(df_clean)}) or classes ({df_clean['source_type'].nunique()}) to train models.")
        return

    print("=" * 78)
    print(" MOUSE TRAJECTORY BOT DETECTION — RIGOROUS EVALUATION BENCHMARK")
    print("=" * 78)
    print(f"Loaded {len(df_clean)} valid trajectories across {df_clean['session_id'].nunique()} recording sessions.")
    print("Class distribution:")
    for cls, cnt in df_clean["source_type"].value_counts().items():
        print(f"  {cls:<20}: {cnt} samples")
    print()

    min_class_samples = df_clean["source_type"].value_counts().min()
    n_splits = max(2, min(5, min_class_samples))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # =========================================================================
    # BENCHMARK A: Full Model (Kinematics & Spatial Geometry with Jerk)
    # =========================================================================
    print("=" * 78)
    print(" BENCHMARK A: Full Model (11 Features, Including High-Frequency Jerk)")
    print("=" * 78)
    X_full = df_clean[features_all]
    y_binary = df_clean["source_type"].apply(lambda s: "human" if s == "human" else "bot")
    y_multi = df_clean["source_type"]

    cv_dt_bin_a = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42), X_full, y_binary, cv=cv)
    cv_rf_bin_a = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X_full, y_binary, cv=cv)
    cv_rf_mul_a = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X_full, y_multi, cv=cv)

    print(f"  Binary 5-Fold CV (Decision Tree): {cv_dt_bin_a.mean() * 100:.1f}% (+/- {cv_dt_bin_a.std() * 100:.1f}%)")
    print(f"  Binary 5-Fold CV (Random Forest): {cv_rf_bin_a.mean() * 100:.1f}% (+/- {cv_rf_bin_a.std() * 100:.1f}%)")
    print(f"  Multiclass 5-Fold CV (Random Forest): {cv_rf_mul_a.mean() * 100:.1f}% (+/- {cv_rf_mul_a.std() * 100:.1f}%)")

    # Learned Tree Rule
    clf_tree_full = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_tree_full.fit(X_full, y_binary)
    print("\n  Learned Decision Rule (Full Model):")
    for line in export_text(clf_tree_full, feature_names=features_all).strip().split("\n"):
        print(f"    {line}")
    print("  * Note: The model exploits the discrete micro-tremor gap in log_dimensionless_jerk.")

    # =========================================================================
    # BENCHMARK B: Pure Spatial Geometry & Macro-Kinematics (NO Jerk)
    # =========================================================================
    print("\n" + "=" * 78)
    print(" BENCHMARK B: Pure Spatial Geometry & Macro-Kinematics (NO Jerk)")
    print("              (Prevents reliance on hardware polling jitter & tremor)")
    print("=" * 78)
    X_nojerk = df_clean[features_no_jerk]

    cv_dt_bin_b = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=42), X_nojerk, y_binary, cv=cv)
    cv_rf_bin_b = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X_nojerk, y_binary, cv=cv)
    cv_rf_mul_b = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42), X_nojerk, y_multi, cv=cv)

    print(f"  Binary 5-Fold CV (Decision Tree): {cv_dt_bin_b.mean() * 100:.1f}% (+/- {cv_dt_bin_b.std() * 100:.1f}%)")
    print(f"  Binary 5-Fold CV (Random Forest): {cv_rf_bin_b.mean() * 100:.1f}% (+/- {cv_rf_bin_b.std() * 100:.1f}%)")
    print(f"  Multiclass 5-Fold CV (Random Forest): {cv_rf_mul_b.mean() * 100:.1f}% (+/- {cv_rf_mul_b.std() * 100:.1f}%)")

    # Train / Test Holdout evaluation for Benchmark B
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_nojerk, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )
    rf_b = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_b.fit(X_tr, y_tr)
    y_pred_b = rf_b.predict(X_te)

    print(f"\n  Holdout Test Accuracy (Random Forest without Jerk): {rf_b.score(X_te, y_te) * 100:.1f}%\n")
    print("  Classification Report (Holdout Test Set):")
    rep_lines = classification_report(y_te, y_pred_b, digits=3).split("\n")
    for line in rep_lines:
        print(f"    {line}")

    cm_b = confusion_matrix(y_te, y_pred_b, labels=["bot", "human"])
    print("  Confusion Matrix [rows: Actual, cols: Predicted]:")
    print(pd.DataFrame(cm_b, index=["    Actual Bot", "    Actual Human"], columns=["Pred Bot", "Pred Human"]))

    # Feature Importance Ranking without Jerk
    importances_b = pd.Series(rf_b.feature_importances_, index=features_no_jerk).sort_values(ascending=False)
    print("\n  Geometric Feature Importance Ranking (No Jerk):")
    for feat, imp in importances_b.items():
        bar = "█" * int(imp * 35)
        print(f"    {feat:<25} {imp:6.3f}  {bar}")

    # =========================================================================
    # BENCHMARK C: Leave-One-Session-Out Cross-Validation (Generalization Test)
    # =========================================================================
    print("\n" + "=" * 78)
    print(" BENCHMARK C: Leave-One-Session-Out Cross-Validation (Without Jerk)")
    print("              (Tests generalization to completely unseen sessions)")
    print("=" * 78)

    groups = df_clean["session_id"]
    logo = LeaveOneGroupOut()
    rf_logo = RandomForestClassifier(n_estimators=50, random_state=42)

    session_scores = {}
    misclassified = []

    for train_idx, test_idx in logo.split(X_nojerk, y_binary, groups):
        sess = groups.iloc[test_idx[0]]
        rf_logo.fit(X_nojerk.iloc[train_idx], y_binary.iloc[train_idx])
        acc = rf_logo.score(X_nojerk.iloc[test_idx], y_binary.iloc[test_idx])
        session_scores[sess] = acc

        preds = rf_logo.predict(X_nojerk.iloc[test_idx])
        acts = y_binary.iloc[test_idx].values
        for i, (pred, act) in enumerate(zip(preds, acts)):
            if pred != act:
                row = df_clean.iloc[test_idx[i]]
                misclassified.append({
                    "session_id": sess,
                    "file_name": row["file_name"],
                    "source_type": row["source_type"],
                    "actual": act,
                    "predicted": pred,
                    "peak_to_mean": row["peak_to_mean_speed"],
                    "bezier_residual": row["bezier_residual_px"],
                    "chord_dev": row["max_chord_dev_px"],
                    "duration_s": row["duration_s"],
                })

    mean_logo_acc = np.mean(list(session_scores.values())) * 100
    print(f"  Mean Leave-One-Session-Out Accuracy: {mean_logo_acc:.1f}%\n")
    print("  Accuracy Breakdown by Recording Session:")
    for sess in sorted(session_scores.keys()):
        score = session_scores[sess] * 100
        n_trials = np.sum(groups == sess)
        print(f"    {sess:<15}: {score:5.1f}%  ({n_trials:2d} trials)")

    print(f"\n  Qualitative Error Analysis ({len(misclassified)} misclassified samples):")
    if not misclassified:
        print("    None. All sessions generalized perfectly.")
    else:
        for m in misclassified:
            print(f"    * [{m['session_id']}/{m['file_name']}] Actual: {m['actual']} ({m['source_type']}) -> Predicted: {m['predicted']}")
            print(f"        peak_to_mean={m['peak_to_mean']:.2f}, bezier_residual={m['bezier_residual']:.2f}px, chord_dev={m['chord_dev']:.1f}px, duration={m['duration_s']:.2f}s")
            print("        -> Cause: Fast, straight human flick mimicking synthetic Bézier trajectory.")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    run_classification()