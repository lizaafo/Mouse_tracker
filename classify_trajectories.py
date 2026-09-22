"""
classify_trajectories.py

שלב 5 בפרויקט: סיווג תנועות עכבר בין אדם לבוטים על בסיס מדדים גאומטריים וקינמטיים.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix


def run_classification():
    csv_path = Path(__file__).resolve().parent / "geometry_summary.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Please run analyze_trajectories.py first.")
        return

    df = pd.read_csv(csv_path)

    # פיצ'רים גאומטריים וקינמטיים ללמידה
    features = [
        "peak_to_mean_speed",
        "max_chord_dev_px",
        "curvature_std",
        "path_ratio",
        "total_angle_change",
    ]

    # סינון שורות תקינות בלבד
    df_clean = df.dropna(subset=features + ["source_type"]).copy()
    print(f"Loaded {len(df_clean)} valid trajectories from {csv_path.name}")
    print(f"Distribution:\n{df_clean['source_type'].value_counts()}\n")

    X = df_clean[features]

    # =========================================================================
    # משימה 1: סיווג בינארי (אדם מול בוט)
    # =========================================================================
    print("=" * 60)
    print(" משימה 1: סיווג בינארי (Human vs. Bot)")
    print("=" * 60)

    y_binary = df_clean["source_type"].apply(lambda s: "human" if s == "human" else "bot")

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )

    clf_binary = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf_binary.fit(X_train_b, y_train_b)

    y_pred_b = clf_binary.predict(X_test_b)
    acc_b = clf_binary.score(X_test_b, y_test_b)

    print(f"Test Accuracy: {acc_b * 100:.1f}%\n")
    print("Classification Report:")
    print(classification_report(y_test_b, y_pred_b))

    print("Confusion Matrix [rows: True, cols: Predicted]:")
    cm_b = confusion_matrix(y_test_b, y_pred_b, labels=["bot", "human"])
    print(pd.DataFrame(cm_b, index=["Actual Bot", "Actual Human"], columns=["Pred Bot", "Pred Human"]))

    print("\nLearned Decision Rules (Binary Tree):")
    print(export_text(clf_binary, feature_names=features))

    # =========================================================================
    # משימה 2: סיווג מרובה מחלקות (זיהוי סוג התנועה הספציפי)
    # =========================================================================
    print("\n" + "=" * 60)
    print(" משימה 2: סיווג מרובה מחלקות (Multiclass: human, linear, curved, noisy)")
    print("=" * 60)

    y_multi = df_clean["source_type"]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=0.3, random_state=42, stratify=y_multi
    )

    clf_multi = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf_multi.fit(X_train_m, y_train_m)

    y_pred_m = clf_multi.predict(X_test_m)
    acc_m = clf_multi.score(X_test_m, y_test_m)

    print(f"Test Accuracy: {acc_m * 100:.1f}%\n")
    print("Classification Report:")
    print(classification_report(y_test_m, y_pred_m))

    classes = sorted(y_multi.unique())
    print("Confusion Matrix:")
    cm_m = confusion_matrix(y_test_m, y_pred_m, labels=classes)
    print(pd.DataFrame(cm_m, index=[f"Actual {c}" for c in classes], columns=[f"Pred {c}" for c in classes]))

    print("\nLearned Decision Rules (Multiclass Tree):")
    print(export_text(clf_multi, feature_names=features))


if __name__ == "__main__":
    run_classification()