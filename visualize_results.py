"""
visualize_results.py

שלב 4 בפרויקט: הפקת גרפים וויזואליזציות להשוואה בין אדם לבוטים ולהגשה בדוח הסופי.
יוצר 4 גרפים ושומר אותם בתיקיית plots/:
1. trajectories_comparison.png
2. features_scatter.png
3. velocity_profiles.png
4. decision_tree_diagram.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
SUMMARY_CSV = BASE_DIR / "geometry_summary.csv"


def setup_plots_dir():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# גרף 1: השוואת 4 מסלולים מייצגים במישור
# =============================================================================
def plot_trajectories_comparison():
    print("Generating Figure 1: Trajectories comparison...")
    # נבחר 4 קבצים מייצגים מתוך session_01
    sample_files = {
        "Human Movement": DATA_DIR / "session_01" / "P02.csv",
        "Linear Bot (Straight)": DATA_DIR / "session_01" / "P06.csv",
        "Curved Bot (Bézier Arc)": DATA_DIR / "session_01" / "P07.csv",
        "Noisy Bot (Perturbed)": DATA_DIR / "session_01" / "P08.csv",
    }

    colors = {
        "Human Movement": "#1f77b4",
        "Linear Bot (Straight)": "#2ca02c",
        "Curved Bot (Bézier Arc)": "#ff7f0e",
        "Noisy Bot (Perturbed)": "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for idx, (title, filepath) in enumerate(sample_files.items()):
        ax = axes[idx]
        if not filepath.exists():
            ax.text(0.5, 0.5, f"File not found:\n{filepath.name}", ha="center", va="center")
            continue

        df = pd.read_csv(filepath)
        start_x = df["start_x"].iloc[0]
        start_y = df["start_y"].iloc[0]
        target_x = df["target_x"].iloc[0]
        target_y = df["target_y"].iloc[0]

        # ציור עיגול התחלה (ירוק) ומטרה (אדום)
        start_circle = plt.Circle((start_x, start_y), 20, color="#4caf50", alpha=0.4, label="Start")
        target_circle = plt.Circle((target_x, target_y), 20, color="#f44336", alpha=0.4, label="Target")
        ax.add_patch(start_circle)
        ax.add_patch(target_circle)

        # קו ישר מקווקו לנקודת ייחוס (מיתר ישיר)
        ax.plot([start_x, target_x], [start_y, target_y], "k--", alpha=0.3, label="Direct Chord")

        # ציור המסלול
        ax.plot(df["x"], df["y"], color=colors[title], lw=2.2, label="Trajectory")
        ax.scatter([df["x"].iloc[0]], [df["y"].iloc[0]], color="#2e7d32", s=40, zorder=5)
        ax.scatter([df["x"].iloc[-1]], [df["y"].iloc[-1]], color="#c62828", s=40, zorder=5)

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_aspect("equal", "datalim")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.invert_yaxis()  # קואורדינטות מסך (0 למעלה)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    plt.suptitle("Comparison of Mouse Trajectories: Human vs. Bots", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "trajectories_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path.name}")


# =============================================================================
# גרף 2: תרשים פיזור (Scatter Plot) עם גבולות ההפרדה
# =============================================================================
def plot_features_scatter():
    print("Generating Figure 2: Features scatter plot...")
    if not SUMMARY_CSV.exists():
        print("  Error: summary CSV not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    df = df.dropna(subset=["max_chord_dev_px", "peak_to_mean_speed", "source_type"])

    plt.figure(figsize=(9, 6))
    
    color_map = {
        "human": ("#1f77b4", "o", "Human"),
        "bot_linear": ("#2ca02c", "s", "Linear Bot"),
        "bot_curved": ("#ff7f0e", "D", "Curved Bot"),
        "bot_noisy": ("#d62728", "^", "Noisy Bot"),
        "bot_smart_jerk": ("#9467bd", "P", "Min-Jerk Bot"),
        "bot_smart_full": ("#8c564b", "X", "Biomechanical Bot"),
    }

    for source, (color, marker, label) in color_map.items():
        subset = df[df["source_type"] == source]
        plt.scatter(
            subset["max_chord_dev_px"],
            subset["peak_to_mean_speed"],
            c=color,
            marker=marker,
            s=70,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.5,
            label=f"{label} (n={len(subset)})",
        )

    # קו החלטה בינארי: אדם מול בוטים
    plt.axhline(2.11, color="#d9534f", linestyle="--", lw=1.8, label="Decision Boundary: Human vs. Bot (y = 2.11)")

    # קווי החלטה לבוטים
    plt.axvline(4.81, color="#5bc0de", linestyle=":", lw=1.5, label="Boundary: Linear vs. Noisy (x = 4.81px)")
    plt.axvline(23.20, color="#f0ad4e", linestyle=":", lw=1.5, label="Boundary: Noisy vs. Curved (x = 23.20px)")

    plt.title("Separation of Trajectories by Kinematic & Geometric Features", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Max Chord Deviation [Straightness] (pixels)", fontsize=11)
    plt.ylabel("Peak-to-Mean Speed Ratio [Velocity Profile]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9, framealpha=0.9)
    plt.tight_layout()

    out_path = PLOTS_DIR / "features_scatter.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path.name}")


# =============================================================================
# גרף 3: פרופילי מהירות לאורך זמן (פעמון מהירות אנושי מול בוט)
# =============================================================================
def plot_velocity_profiles():
    print("Generating Figure 3: Velocity profiles...")
    sample_files = {
        "Human": (DATA_DIR / "session_01" / "P02.csv", "#1f77b4", "-"),
        "Linear Bot": (DATA_DIR / "session_01" / "P06.csv", "#2ca02c", "--"),
        "Curved Bot": (DATA_DIR / "session_01" / "P07.csv", "#ff7f0e", "-."),
        "Noisy Bot": (DATA_DIR / "session_01" / "P08.csv", "#d62728", ":"),
        "Min-Jerk Bot": (DATA_DIR / "session_04" / "P01.csv", "#9467bd", "-"),
        "Biomechanical Bot": (DATA_DIR / "session_04" / "P02.csv", "#8c564b", "-"),
    }

    plt.figure(figsize=(9, 5.5))

    for label, (filepath, color, style) in sample_files.items():
        if not filepath.exists():
            continue
        df = pd.read_csv(filepath)
        t = df["time"].values
        x = df["x"].values
        y = df["y"].values

        # סינון נקודות ללא תנועה
        moving = (np.diff(x, prepend=x[0]) != 0) | (np.diff(y, prepend=y[0]) != 0)
        t_m, x_m, y_m = t[moving], x[moving], y[moving]
        if len(t_m) < 5:
            continue

        # חישוב מהירות ב-20 חלונות זמן אחידים
        t_norm = (t_m - t_m[0]) / (t_m[-1] - t_m[0])
        bin_edges = np.linspace(0, 1, 21)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        speeds = []
        for i in range(20):
            idx = (t_norm >= bin_edges[i]) & (t_norm <= bin_edges[i + 1])
            if np.count_nonzero(idx) >= 2:
                dt = t_m[idx][-1] - t_m[idx][0]
                ds = np.sum(np.hypot(np.diff(x_m[idx]), np.diff(y_m[idx])))
                speeds.append(ds / dt if dt > 0 else 0)
            else:
                speeds.append(0)

        # נרמול מהירות ביחס למהירות הממוצעת להשוואה נקייה
        mean_spd = np.mean(speeds) if np.mean(speeds) > 0 else 1.0
        normalized_speed = np.array(speeds) / mean_spd

        plt.plot(bin_centers * 100, normalized_speed, label=label, color=color, linestyle=style, lw=2.5)

    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Constant Speed (Baseline = 1.0)")
    plt.title("Kinematic Velocity Profiles (Fitts' Law / Minimum Jerk)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Normalized Trajectory Progress (%)", fontsize=11)
    plt.ylabel("Speed Relative to Mean Speed (v / v_mean)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()

    out_path = PLOTS_DIR / "velocity_profiles.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path.name}")


# =============================================================================
# גרף 4: ציור עץ ההחלטה (Decision Tree Diagram)
# =============================================================================
def plot_decision_tree():
    print("Generating Figure 4: Decision tree diagram...")
    if not SUMMARY_CSV.exists():
        print("  Error: summary CSV not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    features = [
        "peak_to_mean_speed",
        "max_chord_dev_px",
        "curvature_std",
        "path_ratio",
        "total_angle_change",
    ]
    df_clean = df.dropna(subset=features + ["source_type"]).copy()

    X = df_clean[features]
    y = df_clean["source_type"]

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_tree(
        clf,
        feature_names=features,
        class_names=sorted(y.unique()),
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )
    plt.title("Learned Geometric & Kinematic Decision Tree", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()

    out_path = PLOTS_DIR / "decision_tree_diagram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path.name}")


def main():
    setup_plots_dir()
    plot_trajectories_comparison()
    plot_features_scatter()
    plot_velocity_profiles()
    plot_decision_tree()
    print(f"\nAll plots generated successfully in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()