"""
live_demo.py

מערכת הדגמה אינטראקטיבית בזמן אמת לזיהוי בוטים לפי תנועת העכבר.
מיועד להצגה, הגנה ומצגת מול מרצה וכיתה (Live Presentation Demo).

מבצע סיווג מיידי ומציג כרטיס חיווי עשיר:
- סיווג בינארי (אדם מול בוט) וזיהוי סוג התנועה הספציפי.
- הצגת המדדים הגאומטריים והקינמטיים המרכזיים שהובילו להחלטה:
  * יחס שיא מהירות לממוצע (פרופיל פעמוני מול קבוע)
  * חוק שני-השלישים (מעריך beta וקורלציה)
  * סטייה מקסימלית מהמיתר (ישרות המסלול)
"""

import math
import random
import time
from pathlib import Path
import tkinter as tk
from tkinter import font

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = BASE_DIR / "geometry_summary.csv"


# =============================================================================
# מודול למידת מכונה וסיווג מקומי (In-Memory Classifier)
# =============================================================================
class TrajectoryClassifier:
    def __init__(self, summary_path=SUMMARY_CSV):
        self.summary_path = Path(summary_path)
        self.is_trained = False
        self.features = [
            "power_law_beta",
            "power_law_r",
            "peak_to_mean_speed",
            "max_chord_dev_px",
            "curvature_std",
            "path_ratio",
            "total_angle_change",
        ]
        self.clf_binary = None
        self.clf_multi = None
        self.train()

    def train(self):
        if not self.summary_path.exists():
            return
        try:
            df = pd.read_csv(self.summary_path)
            clean_df = df.dropna(subset=self.features + ["source_type"]).copy()
            if len(clean_df) < 10:
                return

            X = clean_df[self.features]
            y_binary = clean_df["source_type"].apply(lambda s: "human" if s == "human" else "bot")
            y_multi = clean_df["source_type"]

            self.clf_binary = DecisionTreeClassifier(max_depth=3, random_state=42)
            self.clf_binary.fit(X, y_binary)

            self.clf_multi = DecisionTreeClassifier(max_depth=4, random_state=42)
            self.clf_multi.fit(X, y_multi)

            self.is_trained = True
        except Exception as e:
            print(f"Warning: Could not train demo classifier: {e}")

    def predict(self, metrics):
        if not self.is_trained:
            # Fallback heuristic if dataset wasn't loaded
            is_human = metrics.get("peak_to_mean_speed", 1.0) > 1.95 or metrics.get("power_law_beta", 0.0) > 0.15
            return ("human" if is_human else "bot", "human" if is_human else "bot_naive")

        row = pd.DataFrame([{f: metrics.get(f, 0.0) for f in self.features}])
        # Fill any nan with neutral median
        row = row.fillna(0.0)
        pred_binary = self.clf_binary.predict(row)[0]
        pred_multi = self.clf_multi.predict(row)[0]
        return pred_binary, pred_multi


# =============================================================================
# מודול חילוץ מדדים גאומטריים וקינמטיים מהזיכרון בזמן אמת
# =============================================================================
def extract_live_metrics(trajectory, num_points=100, window=7):
    """Compute geometric and kinematic features directly from in-memory trajectory."""
    if len(trajectory) < 3:
        return {}

    points = np.array([[p["x"], p["y"]] for p in trajectory], dtype=float)
    times = np.array([p["time"] for p in trajectory], dtype=float)

    raw_steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    moving = np.r_[True, raw_steps > 0]
    clean = points[moving]
    clean_t = times[moving]
    segments = raw_steps[raw_steps > 0]

    length = float(np.sum(segments))
    direct = float(np.linalg.norm(points[-1] - points[0]))
    path_ratio = (length / direct) if direct > 0 else 1.0

    # Max chord deviation
    chord_vec = clean[-1] - clean[0]
    chord_len = np.linalg.norm(chord_vec)
    if chord_len > 1e-6:
        diff_start = clean - clean[0]
        cross = np.abs(diff_start[:, 0] * chord_vec[1] - diff_start[:, 1] * chord_vec[0])
        max_chord_dev = float(np.max(cross / chord_len))
    else:
        max_chord_dev = 0.0

    # Total angle change
    if len(clean) >= 2:
        vectors = np.diff(clean, axis=0)
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        change = np.diff(angles)
        wrapped = np.arctan2(np.sin(change), np.cos(change))
        total_angle = float(np.abs(wrapped).sum()) if len(wrapped) else 0.0
    else:
        total_angle = 0.0

    # Duration and speed profile
    duration = float(times[-1] - times[0]) if len(times) >= 2 else 0.0
    mean_speed = (length / duration) if duration > 0 else 0.0

    # Velocity profile: Peak to mean speed across 20 time slices
    dt = np.diff(times)
    unique_times = np.r_[True, dt > 1e-4]
    if duration > 0 and np.sum(unique_times) >= 5:
        time_grid = np.linspace(0, duration, 21)
        grid_x = np.interp(time_grid, times[unique_times] - times[0], points[unique_times, 0])
        grid_y = np.interp(time_grid, times[unique_times] - times[0], points[unique_times, 1])
        slice_speeds = np.hypot(np.diff(grid_x), np.diff(grid_y)) / (duration / 20.0)
        v_mean = np.mean(slice_speeds)
        peak_to_mean_speed = float(np.max(slice_speeds) / v_mean) if v_mean > 0 else 1.0
    else:
        peak_to_mean_speed = 1.0

    # Curvature profile & Two-Thirds Power Law
    mean_curvature = 0.0
    curvature_std = 0.0
    power_law_beta = 0.0
    power_law_r = 0.0

    if length > 0 and len(clean) >= 7:
        try:
            cum_s = np.r_[0.0, np.cumsum(segments)]
            grid_s = np.linspace(0.0, length, num_points)
            sampled = np.column_stack([np.interp(grid_s, cum_s, clean[:, ax]) for ax in (0, 1)])
            ds = grid_s[1] - grid_s[0]

            half = window // 2
            offsets = np.arange(-half, half + 1, dtype=float)
            design = np.column_stack([offsets ** deg for deg in range(4)])
            curvatures = []

            for center in range(half, num_points - half):
                loc = sampled[center - half:center + half + 1] - sampled[center]
                coef = np.linalg.lstsq(design, loc, rcond=None)[0]
                first = coef[1] / ds
                second = 2 * coef[2] / (ds ** 2)
                sp = np.linalg.norm(first)
                if sp > 1e-6:
                    k = abs(first[0] * second[1] - first[1] * second[0]) / (sp ** 3)
                    curvatures.append(k)
                else:
                    curvatures.append(0.0)

            curvs_arr = np.array(curvatures)
            valid_curvs = curvs_arr[np.isfinite(curvs_arr)]
            if len(valid_curvs) > 0:
                mean_curvature = float(np.mean(valid_curvs))
                curvature_std = float(np.std(valid_curvs))

            # Two-Thirds Power Law
            if duration > 0 and len(clean_t) >= 7:
                grid_t = np.interp(grid_s, cum_s, clean_t)
                dt_grid = np.gradient(grid_t)
                v_grid = ds / np.maximum(dt_grid, 1e-6)
                v_mid = v_grid[half:num_points - half]
                valid_pl = (curvs_arr > 5e-5) & (v_mid > 5.0) & np.isfinite(curvs_arr) & np.isfinite(v_mid)
                if np.sum(valid_pl) >= 8:
                    lk = np.log(curvs_arr[valid_pl])
                    lv = np.log(v_mid[valid_pl])
                    poly = np.polyfit(lk, lv, 1)
                    power_law_beta = float(-poly[0])
                    r_mat = np.corrcoef(lk, lv)
                    power_law_r = float(r_mat[0, 1]) if r_mat.shape == (2, 2) else 0.0
        except Exception:
            pass

    return {
        "path_length": length,
        "direct_distance": direct,
        "path_ratio": path_ratio,
        "max_chord_dev_px": max_chord_dev,
        "duration_s": duration,
        "mean_speed_px_s": mean_speed,
        "peak_to_mean_speed": peak_to_mean_speed,
        "total_angle_change": total_angle,
        "mean_curvature": mean_curvature,
        "curvature_std": curvature_std,
        "power_law_beta": power_law_beta,
        "power_law_r": power_law_r,
    }


# =============================================================================
# אפליקציית GUI מלאה להדגמה חיה (Live Demo Application)
# =============================================================================
class LiveDemoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mouse Movement Bot Detection — Live Interactive Demo")
        self.root.geometry("960x780")
        self.root.resizable(False, False)

        # Classifier
        self.classifier = TrajectoryClassifier()

        # Canvas & Target setup
        self.radius = 25
        self.start_x, self.start_y = 120, 260
        self.target_x, self.target_y = 800, 260
        self.recording = False
        self.active_source = "human"
        self.trajectory = []
        self.start_time = None
        self.sampling_job = None
        self.bot_plan = None
        self.BOT_DURATION_S = 1.0

        self.source_colors = {
            "human": "#1976d2",
            "bot_linear": "#388e3c",
            "bot_curved": "#f57c00",
            "bot_noisy": "#00796b",
            "bot_smart_jerk": "#7b1fa2",
            "bot_smart_full": "#5d4037",
        }

        self.source_names_he = {
            "human": "אדם (Human)",
            "bot_linear": "בוט ליניארי (Linear Bot)",
            "bot_curved": "בוט מעוקל (Curved Bézier Bot)",
            "bot_noisy": "בוט רועש (Perturbed Noisy Bot)",
            "bot_smart_jerk": "בוט Min-Jerk (Flash & Hogan)",
            "bot_smart_full": "בוט ביומכני (Biomechanical Tremor Bot)",
        }

        self._build_ui()
        self.randomize_targets()

    def _build_ui(self):
        # 1. Header Frame
        header = tk.Frame(self.root, bg="#263238", pady=8)
        header.pack(fill=tk.X)
        title = tk.Label(
            header,
            text="מערכת הדגמה חיה: זיהוי בוטים לפי גאומטריה וקינמטיקה",
            font=("Helvetica", 15, "bold"),
            fg="white",
            bg="#263238",
        )
        title.pack()
        subtitle = tk.Label(
            header,
            text="הזיזו את העכבר מהעיגול הירוק אל האדום, או בחרו אחד מ-5 הבוטים לבדיקת הסיווג בזמן אמת",
            font=("Helvetica", 10),
            fg="#b0bec5",
            bg="#263238",
        )
        subtitle.pack(pady=1)

        # 2. Drawing Canvas (Packed right under header - completely stable coordinate frame)
        canvas_frame = tk.Frame(self.root, bg="#cfd8dc", bd=1, relief=tk.SUNKEN)
        canvas_frame.pack(padx=15, pady=6)
        self.canvas = tk.Canvas(canvas_frame, width=920, height=440, bg="white", highlightthickness=0)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_start_click)
        self.canvas.bind("<Motion>", self.on_mouse_motion)
        self.canvas.bind("<B1-Motion>", self.on_mouse_motion)
        self.canvas.bind("<Leave>", self.on_leave)

        # 3. Live Decision Card (כרטיס תוצאה מודגש - מקובע מתחת לקנבס בגובה קבוע שלא מזיז שום אלמנט)
        self.card_frame = tk.Frame(self.root, bg="#eceff1", bd=2, relief=tk.SOLID, padx=15, pady=6, height=90)
        self.card_frame.pack(fill=tk.X, padx=15, pady=4)
        self.card_frame.pack_propagate(False)  # מונע כל שינוי גודל או תזוזת פיקסלים בממשק

        self.card_verdict = tk.Label(
            self.card_frame,
            text="ממתין לתנועה... לחצו על העיגול הירוק או בחרו בוט למטה",
            font=("Helvetica", 13, "bold"),
            bg="#eceff1",
            fg="#455a64",
        )
        self.card_verdict.pack(pady=2)

        self.card_details = tk.Label(
            self.card_frame,
            text="מדדים יופיעו כאן בזמן אמת מיד עם סיום התנועה",
            font=("Courier", 10),
            bg="#eceff1",
            fg="#607d8b",
        )
        self.card_details.pack(pady=2)

        # 4. Controls Frame
        controls = tk.Frame(self.root, pady=8)
        controls.pack(fill=tk.X, padx=15)

        lbl_bots = tk.Label(controls, text="הפעלת בוטים להדגמה:", font=("Helvetica", 11, "bold"))
        lbl_bots.pack(side=tk.LEFT, padx=6)

        self.bot_buttons = []
        bot_defs = [
            ("בוט ליניארי", lambda: self.run_bot("bot_linear"), "#2ca02c"),
            ("בוט מעוקל", lambda: self.run_bot("bot_curved"), "#ff7f0e"),
            ("בוט רועש", lambda: self.run_bot("bot_noisy"), "#00796b"),
            ("בוט Min-Jerk", lambda: self.run_bot("bot_smart_jerk"), "#9467bd"),
            ("בוט ביומכני", lambda: self.run_bot("bot_smart_full"), "#8c564b"),
        ]

        for text, cmd, col in bot_defs:
            btn = tk.Button(controls, text=text, command=cmd, fg=col, font=("Helvetica", 10, "bold"), padx=5, pady=3)
            btn.pack(side=tk.LEFT, padx=3)
            self.bot_buttons.append(btn)

        # Reset & Randomize buttons
        btn_rnd = tk.Button(controls, text="🎲 הגרלת מטרות", command=self.randomize_targets, font=("Helvetica", 10))
        btn_rnd.pack(side=tk.RIGHT, padx=4)

        self.btn_cancel = tk.Button(controls, text="ביטול", state=tk.DISABLED, command=self.cancel_trial, font=("Helvetica", 10))
        self.btn_cancel.pack(side=tk.RIGHT, padx=4)

    # =========================================================================
    # גאומטריה של המטרות
    # =========================================================================
    def randomize_targets(self):
        if self.recording:
            return
        w, h = 920, 440
        pad = self.radius + 30
        for _ in range(150):
            sx, sy = random.randint(pad, w - pad), random.randint(pad, h - pad)
            tx, ty = random.randint(pad, w - pad), random.randint(pad, h - pad)
            if math.hypot(tx - sx, ty - sy) >= 280:
                self.start_x, self.start_y = sx, sy
                self.target_x, self.target_y = tx, ty
                break

        self.canvas.delete("all")
        # Draw start circle (Green) and target circle (Red)
        self.canvas.create_oval(
            self.start_x - self.radius, self.start_y - self.radius,
            self.start_x + self.radius, self.start_y + self.radius,
            fill="#4caf50", outline="#2e7d32", width=2, tags="targets"
        )
        self.canvas.create_text(self.start_x, self.start_y, text="START", fill="white", font=("Helvetica", 8, "bold"), tags="targets")

        self.canvas.create_oval(
            self.target_x - self.radius, self.target_y - self.radius,
            self.target_x + self.radius, self.target_y + self.radius,
            fill="#f44336", outline="#c62828", width=2, tags="targets"
        )
        self.canvas.create_text(self.target_x, self.target_y, text="TARGET", fill="white", font=("Helvetica", 8, "bold"), tags="targets")

        # Chord line (dashed)
        self.canvas.create_line(self.start_x, self.start_y, self.target_x, self.target_y, fill="#cfd8dc", dash=(3, 3), tags="targets")

    def inside_circle(self, x, y, cx, cy, r):
        return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2

    def segment_intersects_circle(self, x1, y1, x2, y2, cx, cy, r):
        dx, dy = x2 - x1, y2 - y1
        l_sq = dx ** 2 + dy ** 2
        if l_sq == 0:
            return self.inside_circle(x1, y1, cx, cy, r)
        t = max(0.0, min(1.0, ((cx - x1) * dx + (cy - y1) * dy) / l_sq))
        return self.inside_circle(x1 + t * dx, y1 + t * dy, cx, cy, r)

    # =========================================================================
    # תנועת משתמש אנושי
    # =========================================================================
    def on_start_click(self, event):
        if self.recording:
            return
        if not self.inside_circle(event.x, event.y, self.start_x, self.start_y, self.radius):
            return
        self.begin_movement("human", event.x, event.y)

    def on_mouse_motion(self, event):
        if not self.recording or self.active_source != "human":
            return
        elapsed = time.perf_counter() - self.start_time
        self.add_point(event.x, event.y, elapsed)

    def on_leave(self, event):
        if self.recording and self.active_source == "human":
            self.cancel_trial(reason="הסמן יצא מגבולות המשטח")

    # =========================================================================
    # הפעלת בוטים
    # =========================================================================
    def make_bot_plan(self, source_type, seed):
        rng = random.Random(seed)
        angle = rng.uniform(0, 2 * math.pi)
        offset = self.radius * 0.7 * math.sqrt(rng.random())
        start = (round(self.start_x + offset * math.cos(angle)), round(self.start_y + offset * math.sin(angle)))
        target = (self.target_x, self.target_y)
        dx, dy = target[0] - start[0], target[1] - start[1]
        dist = math.hypot(dx, dy)
        normal = (-dy / dist, dx / dist)
        midpoint = ((start[0] + target[0]) / 2, (start[1] + target[1]) / 2)

        plan = {
            "source_type": source_type,
            "start": start,
            "target": target,
            "normal": normal,
            "duration": self.BOT_DURATION_S,
            "control": None,
            "noise_knots": [],
            "tremor_phases": (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi)),
        }

        if source_type in ("bot_curved", "bot_smart_jerk", "bot_smart_full"):
            sign = rng.choice([-1, 1])
            curve_ratio = 0.25 if source_type == "bot_smart_full" else 0.35
            amplitude = dist * curve_ratio * sign
            plan["control"] = (midpoint[0] + amplitude * normal[0], midpoint[1] + amplitude * normal[1])
        elif source_type == "bot_noisy":
            amp = min(12.0, dist * 0.03)
            plan["noise_knots"] = [0.0] + [rng.uniform(-amp, amp) for _ in range(7)] + [0.0]

        return plan

    def evaluate_bot(self, plan, elapsed):
        tau = min(1.0, max(0.0, elapsed / plan["duration"]))
        start, target = plan["start"], plan["target"]

        if plan["source_type"] == "bot_smart_jerk":
            u = 10 * (tau ** 3) - 15 * (tau ** 4) + 6 * (tau ** 5)
            ctrl = plan["control"]
            return ((1 - u) ** 2 * start[0] + 2 * (1 - u) * u * ctrl[0] + u * u * target[0],
                    (1 - u) ** 2 * start[1] + 2 * (1 - u) * u * ctrl[1] + u * u * target[1])

        if plan["source_type"] == "bot_smart_full":
            tau_w = tau ** 0.8
            u = 10 * (tau_w ** 3) - 15 * (tau_w ** 4) + 6 * (tau_w ** 5)
            u = min(1.0, max(0.0, u))
            ctrl = plan["control"]
            bx = (1 - u) ** 2 * start[0] + 2 * (1 - u) * u * ctrl[0] + u * u * target[0]
            by = (1 - u) ** 2 * start[1] + 2 * (1 - u) * u * ctrl[1] + u * u * target[1]
            env = 4.0 * u * (1.0 - u)
            phi1, phi2 = plan["tremor_phases"]
            tremor = 2.5 * env * (0.7 * math.sin(2 * math.pi * 10.0 * elapsed + phi1) +
                                  0.3 * math.cos(2 * math.pi * 18.0 * elapsed + phi2))
            return (bx + tremor * plan["normal"][0], by + tremor * plan["normal"][1])

        u = tau
        if plan["source_type"] == "bot_curved":
            ctrl = plan["control"]
            return ((1 - u) ** 2 * start[0] + 2 * (1 - u) * u * ctrl[0] + u * u * target[0],
                    (1 - u) ** 2 * start[1] + 2 * (1 - u) * u * ctrl[1] + u * u * target[1])

        base_x = start[0] + u * (target[0] - start[0])
        base_y = start[1] + u * (target[1] - start[1])

        if plan["source_type"] == "bot_noisy":
            knots = plan["noise_knots"]
            pos = u * (len(knots) - 1)
            idx = min(int(pos), len(knots) - 2)
            frac = pos - idx
            weight = frac * frac * (3 - 2 * frac)
            offset = (1 - weight) * knots[idx] + weight * knots[idx + 1]
            return (base_x + offset * plan["normal"][0], base_y + offset * plan["normal"][1])

        return (base_x, base_y)

    def run_bot(self, source_type):
        if self.recording:
            return
        seed = random.randint(1, 1000000)
        self.bot_plan = self.make_bot_plan(source_type, seed)
        self.begin_movement(source_type, *self.bot_plan["start"])
        self.bot_step()

    def bot_step(self):
        if not self.recording or self.active_source == "human":
            return
        elapsed = time.perf_counter() - self.start_time
        x, y = self.evaluate_bot(self.bot_plan, elapsed)
        self.add_point(round(x), round(y), elapsed)

        if self.recording:
            self.sampling_job = self.root.after(5, self.bot_step)

    # =========================================================================
    # רישום וניהול תנועה
    # =========================================================================
    def begin_movement(self, source_type, start_x, start_y):
        self.recording = True
        self.active_source = source_type
        self.canvas.delete("path")
        self.trajectory = [{"x": start_x, "y": start_y, "time": 0.0}]
        self.start_time = time.perf_counter()

        for b in self.bot_buttons:
            b.config(state=tk.DISABLED)
        self.btn_cancel.config(state=tk.NORMAL)

        # Update card
        self.card_frame.config(bg="#e0f7fa", bd=2)
        self.card_verdict.config(
            text="מקליט תנועה בזמן אמת... מעקב פעיל",
            fg="#006064",
            bg="#e0f7fa",
        )
        self.card_details.config(text="תנועה פעילה לעבר המטרה האדומה...", bg="#e0f7fa")

    def add_point(self, x, y, elapsed):
        if not self.recording:
            return
        prev = self.trajectory[-1]
        if x == prev["x"] and y == prev["y"]:
            return
        self.trajectory.append({"x": x, "y": y, "time": elapsed})

        # Draw line segment
        self.canvas.create_line(
            prev["x"], prev["y"], x, y,
            fill=self.source_colors.get(self.active_source, "blue"),
            width=2.5, tags="path"
        )

        # Check target arrival
        if self.inside_circle(x, y, self.target_x, self.target_y, self.radius) or \
           self.segment_intersects_circle(prev["x"], prev["y"], x, y, self.target_x, self.target_y, self.radius):
            self.finish_movement()

    def cancel_trial(self, reason="התנועה בוטלה"):
        self.recording = False
        if self.sampling_job:
            self.root.after_cancel(self.sampling_job)
            self.sampling_job = None
        self.canvas.delete("path")
        for b in self.bot_buttons:
            b.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)

        self.card_frame.config(bg="#fbe9e7", bd=2)
        self.card_verdict.config(text=f"ניסיון בוטל: {reason}", fg="#d84315", bg="#fbe9e7")
        self.card_details.config(text="לחצו על העיגול הירוק או בחרו בוט לניסיון חדש", bg="#fbe9e7")

    # =========================================================================
    # סיום תנועה וסיווג חי
    # =========================================================================
    def finish_movement(self):
        self.recording = False
        if self.sampling_job:
            self.root.after_cancel(self.sampling_job)
            self.sampling_job = None

        for b in self.bot_buttons:
            b.config(state=tk.NORMAL)
        self.btn_cancel.config(state=tk.DISABLED)

        # 1. חילוץ מדדים מהיר
        metrics = extract_live_metrics(self.trajectory)

        # 2. סיווג בעץ ההחלטה
        pred_bin, pred_multi = self.classifier.predict(metrics)

        # 3. עדכון כרטיס תוצאה חי
        self.display_live_verdict(pred_bin, pred_multi, metrics)

    def display_live_verdict(self, pred_bin, pred_multi, m):
        is_human = (pred_bin == "human")
        actual = self.active_source
        actual_name = self.source_names_he.get(actual, actual)
        pred_name = self.source_names_he.get(pred_multi, pred_multi)

        if is_human:
            bg_col = "#e8f5e9"
            border_col = "#2e7d32"
            title_text = "🟢 תוצאת סיווג: 👤 זוהתה תנועה אנושית! (Human Movement)"
            fg_col = "#1b5e20"
        else:
            bg_col = "#ffebee"
            border_col = "#c62828"
            title_text = f"🔴 תוצאת סיווג: 🤖 זוהה בוט אוטומטי! [{pred_name}]"
            fg_col = "#b71c1c"

        # Format details
        p2m = m.get("peak_to_mean_speed", 1.0)
        beta = m.get("power_law_beta", 0.0)
        chord = m.get("max_chord_dev_px", 0.0)
        k_mean = m.get("mean_curvature", 0.0)
        dur = m.get("duration_s", 0.0)

        details_text = (
            f"מקור אמיתי: {actual_name} | זמן תנועה: {dur:.2f}s\n"
            f"יחס שיא/ממוצע מהירות: {p2m:.2f} | מעריך חוק שני-השלישים (β): {beta:+.3f} | "
            f"סטיית מיתר: {chord:.1f}px | עקמומיות ממוצעת: {k_mean:.4f}"
        )

        self.card_frame.config(bg=bg_col, bd=2)
        self.card_verdict.config(text=title_text, fg=fg_col, bg=bg_col)
        self.card_details.config(text=details_text, fg="#37474f", bg=bg_col)


def main():
    root = tk.Tk()
    app = LiveDemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
