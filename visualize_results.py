"""
visualize_results.py

שלב 4 בפרויקט: הפקת גרפים וויזואליזציות להשוואה בין אדם לבוטים ולהגשה בדוח הסופי.
יוצר 4 גרפים דינמיים ושומר אותם בתיקיית plots/:
1. trajectories_comparison.png - השוואת מסלולים מייצגים במישור
2. features_scatter.png - תרשים פיזור עם קווי החלטה דינמיים
3. velocity_profiles.png - פרופילי מהירות לאורך זמן (Fitts' Law / Minimum Jerk)
4. decision_tree_diagram.png - תרשים עץ ההחלטה הגיאומטרי והקינמטי
"""

import csv
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


def find_sample_trajectory(source_type):
    """Dynamically find the first valid trajectory CSV file for a given source type."""
    if not DATA_DIR.exists():
        return None
    for csv_file in sorted(DATA_DIR.glob("session_*/*.csv")):
        try:
            with csv_file.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row and row.get("source_type", "human") == source_type:
                    return csv_file
        except Exception:
            continue
    return None


# =============================================================================
# גרף 1: השוואת 4 מסלולים מייצגים במישור
# =============================================================================
def plot_trajectories_comparison():
    print("Generating Figure 1: Trajectories comparison...")
    
    sources = [
        ("Human Movement", "human", "#1f77b4"),
        ("Linear Bot (Straight)", "bot_linear", "#2ca02c"),
        ("Curved Bot (Bézier)", "bot_curved", "#ff7f0e"),
        ("Biomechanical Bot", "bot_smart_full", "#8c564b"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for idx, (title, source_type, color) in enumerate(sources):
        ax = axes[idx]
        filepath = find_sample_trajectory(source_type)
        if not filepath:
            # Fallback to noisy bot if smart_full not available
            if source_type == "bot_smart_full":
                filepath = find_sample_trajectory("bot_noisy")
                title = "Noisy Bot (Perturbed)"
                color = "#d62728"

        if not filepath or not filepath.exists():
            ax.text(0.5, 0.5, f"No samples found for:\n{title}", ha="center", va="center")
            continue

        df = pd.read_csv(filepath)
        start_x = df["start_x"].iloc[0]
        start_y = df["start_y"].iloc[0]
        target_x = df["target_x"].iloc[0]
        target_y = df["target_y"].iloc[0]

        # ציור עיגול התחלה (ירוק) ומטרה (אדום)
        start_circle = plt.Circle((start_x, start_y), 25, color="#4caf50", alpha=0.35, label="Start Circle")
        target_circle = plt.Circle((target_x, target_y), 25, color="#f44336", alpha=0.35, label="Target Circle")
        ax.add_patch(start_circle)
        ax.add_patch(target_circle)

        # קו ישר מקווקו לנקודת ייחוס (מיתר ישיר)
        ax.plot([start_x, target_x], [start_y, target_y], "k--", alpha=0.3, label="Direct Chord")

        # ציור המסלול
        ax.plot(df["x"], df["y"], color=color, lw=2.2, label=f"Path ({filepath.parent.name}/{filepath.stem})")
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
# גרף 2: תרשים פיזור (Scatter Plot) עם גבולות החלטה דינמיים
# =============================================================================
def plot_features_scatter():
    print("Generating Figure 2: Features scatter plot with dynamic boundaries...")
    if not SUMMARY_CSV.exists():
        print("  Error: summary CSV not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    df = df.dropna(subset=["max_chord_dev_px", "peak_to_mean_speed", "source_type"])
    if df.empty:
        print("  Error: no valid data in summary CSV.")
        return

    plt.figure(figsize=(9.5, 6.5))

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
        if len(subset) == 0:
            continue
        plt.scatter(
            subset["max_chord_dev_px"],
            subset["peak_to_mean_speed"],
            c=color,
            marker=marker,
            s=75,
            alpha=0.85,
            edgecolors="k",
            linewidths=0.6,
            label=f"{label} (n={len(subset)})",
        )

    # חילוץ דינמי של קו ההחלטה הבינארי (אדם מול בוט) מתוך עץ החלטה
    try:
        y_bin = df["source_type"].apply(lambda s: 1 if s == "human" else 0)
        if y_bin.nunique() >= 2:
            clf_bin = DecisionTreeClassifier(max_depth=1, random_state=42)
            clf_bin.fit(df[["peak_to_mean_speed"]], y_bin)
            v_thresh = clf_bin.tree_.threshold[0]
            if v_thresh != -2:
                plt.axhline(
                    v_thresh,
                    color="#d9534f",
                    linestyle="--",
                    lw=1.8,
                    label=f"Learned Boundary: Human vs Bot (y = {v_thresh:.2f})",
                )
    except Exception as e:
        print(f"  Note on binary threshold: {e}")

    # חילוץ דינמי של גבולות ישרות עבור הבוטים הנאיביים (Linear vs Noisy vs Curved)
    try:
        df_naive = df[df["source_type"].isin(["bot_linear", "bot_noisy", "bot_curved"])]
        if not df_naive.empty and df_naive["source_type"].nunique() >= 2:
            clf_chord = DecisionTreeClassifier(max_depth=2, random_state=42)
            clf_chord.fit(df_naive[["max_chord_dev_px"]], df_naive["source_type"])
            chord_threshs = sorted([t for t in clf_chord.tree_.threshold if t != -2])
            line_styles = [(":", "#5bc0de"), (":", "#f0ad4e")]
            for idx, th in enumerate(chord_threshs[:2]):
                style, col = line_styles[idx]
                plt.axvline(
                    th,
                    color=col,
                    linestyle=style,
                    lw=1.5,
                    label=f"Learned Shape Boundary (x = {th:.2f}px)",
                )
    except Exception as e:
        print(f"  Note on chord threshold: {e}")

    plt.title("Dynamic Separation of Trajectories by Kinematic & Geometric Features", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Max Chord Deviation [Straightness] (pixels)", fontsize=11)
    plt.ylabel("Peak-to-Mean Speed Ratio [Velocity Profile]", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.55)
    plt.legend(loc="upper right", fontsize=8.5, framealpha=0.92)
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
    sources = [
        ("Human", "human", "#1f77b4", "-"),
        ("Linear Bot", "bot_linear", "#2ca02c", "--"),
        ("Curved Bot", "bot_curved", "#ff7f0e", "-."),
        ("Noisy Bot", "bot_noisy", "#d62728", ":"),
        ("Min-Jerk Bot", "bot_smart_jerk", "#9467bd", "-"),
        ("Biomechanical Bot", "bot_smart_full", "#8c564b", "-"),
    ]

    plt.figure(figsize=(9.5, 5.5))

    plotted_count = 0
    for label, source_type, color, style in sources:
        filepath = find_sample_trajectory(source_type)
        if not filepath or not filepath.exists():
            continue

        try:
            df = pd.read_csv(filepath)
            t = df["time"].values
            x = df["x"].values
            y = df["y"].values

            # סינון נקודות ללא תנועה
            moving = (np.diff(x, prepend=x[0]) != 0) | (np.diff(y, prepend=y[0]) != 0)
            t_m, x_m, y_m = t[moving], x[moving], y[moving]
            if len(t_m) < 5 or (t_m[-1] - t_m[0]) <= 0:
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

            mean_spd = np.mean(speeds) if np.mean(speeds) > 0 else 1.0
            normalized_speed = np.array(speeds) / mean_spd

            plt.plot(bin_centers * 100, normalized_speed, label=label, color=color, linestyle=style, lw=2.4)
            plotted_count += 1
        except Exception as err:
            print(f"  Skipping {label} velocity profile: {err}")

    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7, label="Constant Speed (Baseline = 1.0)")
    plt.title("Kinematic Velocity Profiles (Fitts' Law / Minimum Jerk)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Normalized Trajectory Progress (%)", fontsize=11)
    plt.ylabel("Speed Relative to Mean Speed (v / v_mean)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right", fontsize=9.5)
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
    if len(df_clean) < 5:
        print("  Insufficient clean samples for decision tree plot.")
        return

    X = df_clean[features]
    y = df_clean["source_type"]

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(12, 7))
    plot_tree(
        clf,
        feature_names=features,
        class_names=[str(c) for c in sorted(y.unique())],
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


# =============================================================================
# גרף 5: חוק שני-השלישים (Two-Thirds Power Law Analysis)
# =============================================================================
def plot_two_thirds_power_law():
    print("Generating Figure 5: Two-Thirds Power Law analysis...")
    if not SUMMARY_CSV.exists():
        print("  Error: summary CSV not found.")
        return

    df = pd.read_csv(SUMMARY_CSV)
    df_clean = df.dropna(subset=["power_law_beta", "source_type"])
    if df_clean.empty:
        print("  Error: no power law data found in summary CSV.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # פאנל שמאלי: Boxplot של מעריך חוק שני השלישים (Beta) לפי סוג תנועה
    sources_order = ["human", "bot_linear", "bot_curved", "bot_noisy", "bot_smart_jerk", "bot_smart_full"]
    labels_order = ["Human", "Linear", "Curved", "Noisy", "Min-Jerk", "Biomechanical"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    data_to_plot = []
    active_labels = []
    active_colors = []
    for src, lbl, col in zip(sources_order, labels_order, colors):
        vals = df_clean[df_clean["source_type"] == src]["power_law_beta"].values
        if len(vals) > 0:
            data_to_plot.append(vals)
            active_labels.append(f"{lbl}\n(n={len(vals)})")
            active_colors.append(col)

    bplot = ax1.boxplot(data_to_plot, tick_labels=active_labels, patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for patch, col in zip(bplot['boxes'], active_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)

    ax1.axhline(0.333, color="green", linestyle="--", lw=1.5, label=r"Theoretical $\beta = 1/3$ (Two-Thirds Law)")
    ax1.axhline(0.0, color="gray", linestyle=":", lw=1.2, label=r"Zero Coupling ($\beta = 0$)")
    ax1.set_title("Power Law Exponent (Beta) Across Movement Sources", fontsize=12, fontweight="bold")
    ax1.set_ylabel(r"Power Law Exponent [$\beta$]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.55)
    ax1.legend(loc="upper right", fontsize=8.5)

    # פאנל ימני: פיזור log(kappa) מול log(v) של תנועה אנושית מייצגת מול בוט
    human_file = find_sample_trajectory("human")
    bot_file = find_sample_trajectory("bot_smart_full") or find_sample_trajectory("bot_curved")

    for file_p, label, col in [(human_file, "Human Movement", "#1f77b4"), (bot_file, "Bot Movement", "#8c564b")]:
        if not file_p or not file_p.exists():
            continue
        try:
            d = pd.read_csv(file_p)
            pts = np.column_stack([d['x'], d['y']])
            times = d['time'].values
            dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            m = np.r_[True, dists > 0]
            cp, ct = pts[m], times[m]
            cs = np.r_[0, np.cumsum(dists[dists > 0])]
            if cs[-1] == 0:
                continue
            gs = np.linspace(0, cs[-1], 100)
            gt = np.interp(gs, cs, ct)
            ds = gs[1] - gs[0]
            dt = np.gradient(gt)
            v = ds / np.maximum(dt, 1e-6)

            offsets = np.arange(-3, 4, dtype=float)
            design = np.column_stack([offsets ** deg for deg in range(4)])
            sx = np.interp(gs, cs, cp[:, 0])
            sy = np.interp(gs, cs, cp[:, 1])
            s_pts = np.column_stack([sx, sy])
            curvs = []
            for c in range(3, 97):
                loc = s_pts[c - 3:c + 4] - s_pts[c]
                coef = np.linalg.lstsq(design, loc, rcond=None)[0]
                f = coef[1] / ds
                s = 2 * coef[2] / (ds**2)
                sp = np.linalg.norm(f)
                k = abs(f[0]*s[1] - f[1]*s[0]) / (sp**3) if sp > 1e-5 else 0
                curvs.append(k)
            curvs = np.array(curvs)
            v_mid = v[3:97]
            valid = (curvs > 1e-4) & (v_mid > 5.0)
            if np.sum(valid) >= 10:
                lk = np.log(curvs[valid])
                lv = np.log(v_mid[valid])
                ax2.scatter(lk, lv, color=col, alpha=0.6, s=35, label=f"{label} points")
                p = np.polyfit(lk, lv, 1)
                ax2.plot(lk, np.polyval(p, lk), color=col, lw=2.2, label=rf"{label} Fit ($\beta={-p[0]:.2f}$)")
        except Exception:
            pass

    ax2.set_title(r"Tangential Speed vs. Curvature: $\log(v) = C - \beta\log(\kappa)$", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Log Curvature [$\ln(\kappa)$]", fontsize=11)
    ax2.set_ylabel(r"Log Tangential Speed [$\ln(v)$]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.55)
    ax2.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Validation of the Two-Thirds Power Law in Mouse Movements", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = PLOTS_DIR / "two_thirds_power_law.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved: {out_path.name}")


def main():
    setup_plots_dir()
    plot_trajectories_comparison()
    plot_features_scatter()
    plot_velocity_profiles()
    plot_decision_tree()
    plot_two_thirds_power_law()
    print(f"\nAll plots generated successfully in: {PLOTS_DIR}")


if __name__ == "__main__":
    main()