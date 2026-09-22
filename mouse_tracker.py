# Record mouse trajectories from the green start circle to the red target.
# Each completed trial is saved as CSV; cancelled trials are discarded.

# Tkinter provides the interface; the other modules handle files, timing and IDs.
import tkinter as tk
import csv
import math
import random
import time
import uuid
from pathlib import Path


# Create the main window
root = tk.Tk()
root.title("Mouse Movement Tracker")


# Create the drawing canvas
canvas = tk.Canvas(
    root,
    width=900,
    height=600,
    background="white"
)
# Keep the canvas anchored when status text or the window width changes.
canvas.pack(anchor="nw")


# Circle centers and radius in canvas pixels. The origin is the top-left corner.
# X increases to the right; Y increases downward.
start_x, start_y = 100, 300
target_x, target_y = 800, 300
radius = 25
# Leave space around both circles and avoid very short trials.
CIRCLE_MARGIN = 20
MIN_CENTER_DISTANCE = 250


# Shared trial state used by the mouse callbacks.
# trajectory stores ordered dictionaries containing x, y and elapsed time.
recording = False
trajectory = []
start_time = None
# Start at P01 and advance only after a trajectory is saved successfully.
participant_number = 1
participant_id = f"P{participant_number:02d}"

# Requested delay between polling callbacks, not a guaranteed sampling period.
SAMPLE_INTERVAL_MS = 5
# Keep the scheduled callback ID so it can be cancelled when a trial ends.
sampling_job = None
# Store canvas dimensions and the requested interval for the current trial.
trial_metadata = {}
# Only one recording owns trajectory at a time; buttons and clicks share this guard.
active_source = "human"
bot_plan = None
BOT_DURATION_S = 1.0  # Common baseline for all three bots, not a measured human duration.
SOURCE_COLORS = {"human": "blue", "bot_linear": "purple",
                 "bot_curved": "orange", "bot_noisy": "teal",
                 "bot_smart_jerk": "magenta", "bot_smart_full": "brown"}

# Create the data directory next to this script
DATA_DIRECTORY = Path(__file__).resolve().parent / "data"
DATA_DIRECTORY.mkdir(exist_ok=True)

# Each application launch creates a new session and preserves all earlier data.
def create_session_directory(data_directory):
    # Continue after the highest existing session number, including gaps.
    numbers = [
        int(path.name[8:])
        for path in data_directory.iterdir()
        if path.name.startswith("session_") and path.name[8:].isdigit()
    ]
    number = max(numbers, default=0) + 1
    while True:
        directory = data_directory / f"session_{number:02d}"
        try:
            # Exclusive creation also prevents collisions between simultaneous launches.
            directory.mkdir()
            return directory
        except FileExistsError:
            number += 1


SESSION_DIRECTORY = create_session_directory(DATA_DIRECTORY)
SESSION_ID = SESSION_DIRECTORY.name
root.title(f"Mouse Movement Tracker — {SESSION_ID}")
print(f"Session data directory: {SESSION_DIRECTORY}")


# Display the automatically assigned participant code
participant_frame = tk.Frame(root)
participant_frame.pack(pady=5)

participant_label = tk.Label(
    participant_frame,
    text=f"מזהה משתתף: {participant_id}"
)
participant_label.pack(side=tk.LEFT, padx=5)


# Reserve a stable message area so longer messages do not shift the canvas.
status_label = tk.Label(
    root,
    text="לחצי על העיגול הירוק כדי להתחיל",
    wraplength=860,
    height=3,
    justify="center"
)
status_label.pack()


# Select random centers with a clear gap between the circles.
def choose_circle_positions(width, height):
    padding = radius + CIRCLE_MARGIN
    left, top = padding, padding
    right, bottom = width - padding - 1, height - padding - 1
    if right <= left or bottom <= top:
        raise ValueError("Canvas is too small for the circles")

    # Adapt the distance if the drawing area is smaller than usual.
    diagonal = ((right - left) ** 2 + (bottom - top) ** 2) ** 0.5
    minimum = min(MIN_CENTER_DISTANCE, diagonal * 0.5)
    if minimum <= 2 * radius:
        raise ValueError("Canvas is too small to separate the circles")

    # A bounded loop avoids an indefinite search for a valid layout.
    for _ in range(200):
        sx, sy = random.randint(left, right), random.randint(top, bottom)
        tx, ty = random.randint(left, right), random.randint(top, bottom)
        if (tx - sx) ** 2 + (ty - sy) ** 2 >= minimum ** 2:
            return sx, sy, tx, ty

    # Opposite corners provide a valid fallback if random attempts fail.
    return left, top, right, bottom


# Move both circles only between trials, never during a recording.
def randomize_circles():
    global start_x, start_y, target_x, target_y

    if recording:
        return

    previous = (start_x, start_y, target_x, target_y)
    positions = choose_circle_positions(canvas.winfo_width(), canvas.winfo_height())
    if positions == previous:
        # Swapping distinct centers guarantees a different consecutive layout.
        positions = positions[2:] + positions[:2]
    start_x, start_y, target_x, target_y = positions

    canvas.delete("trial_circles")
    for x, y, color in ((start_x, start_y, "green"), (target_x, target_y, "red")):
        canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius,
            fill=color, tags="trial_circles"
        )


# Include the circle boundary; squared distances avoid an unnecessary square root.
def inside_circle(x, y, center_x, center_y, circle_radius):
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    return distance_squared <= circle_radius ** 2

# Check if the line segment between (x1, y1) and (x2, y2) intersects the circle.
def segment_intersects_circle(x1, y1, x2, y2, center_x, center_y, circle_radius):
    dx = x2 - x1
    dy = y2 - y1
    segment_len_sq = dx ** 2 + dy ** 2
    if segment_len_sq == 0:
        return inside_circle(x1, y1, center_x, center_y, circle_radius)

    # Project circle center onto the segment: t is the normalized parameter in [0, 1]
    t = ((center_x - x1) * dx + (center_y - y1) * dy) / segment_len_sq
    t_clamped = max(0.0, min(1.0, t))
    closest_x = x1 + t_clamped * dx
    closest_y = y1 + t_clamped * dy

    return inside_circle(closest_x, closest_y, center_x, center_y, circle_radius)


# Construct a reproducible geometric plan without touching recording state or files.
def make_bot_plan(source_type, center, target, width, height, seed):
    if source_type not in ("bot_linear", "bot_curved", "bot_noisy", "bot_smart_jerk", "bot_smart_full"):
        raise ValueError("Unknown bot type")
    rng = random.Random(seed)
    # Start inside the green circle, rather than always at its exact center.
    angle = rng.uniform(0, 2 * math.pi)
    offset = radius * 0.8 * math.sqrt(rng.random())
    start = (round(center[0] + offset * math.cos(angle)),
             round(center[1] + offset * math.sin(angle)))
    dx, dy = target[0] - start[0], target[1] - start[1]
    distance = math.hypot(dx, dy)
    if distance <= 2 * radius:
        raise ValueError("Bot start and target must be separated")
    normal = (-dy / distance, dx / distance)
    midpoint = ((start[0] + target[0]) / 2, (start[1] + target[1]) / 2)
    plan = {"source_type": source_type, "start": start, "target": target,
            "normal": normal, "duration": BOT_DURATION_S, "seed": seed,
            "control": None, "noise_amplitude": 0.0, "noise_knots": [],
            "generator_version": "bots_v1"}
    if source_type in ("bot_curved", "bot_smart_jerk", "bot_smart_full"):
        # Check available space in both perpendicular directions (+1 and -1)
        valid_options = []
        for sign in (-1, 1):
            direction = (sign * normal[0], sign * normal[1])
            limits = []
            for coordinate, component, upper in zip(midpoint, direction, (width - 6, height - 6)):
                if component > 1e-12:
                    limits.append((upper - coordinate) / component)
                elif component < -1e-12:
                    limits.append((5 - coordinate) / component)
            max_limit = min(limits) if limits else 0.0
            if max_limit > 5.0:  # At least 5 pixels of clearance
                valid_options.append((sign, direction, max_limit))

        if valid_options:
            sign, direction, max_limit = rng.choice(valid_options)
            curve_range = (0.15, 0.35) if source_type == "bot_smart_full" else (0.2, 0.45)
            amplitude = min(distance * rng.uniform(*curve_range), max_limit * 0.9)
            plan["control"] = tuple(midpoint[i] + amplitude * direction[i] for i in (0, 1))
        else:
            plan["control"] = midpoint

        if source_type == "bot_smart_full":
            plan["tremor_phases"] = (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi))
    elif source_type == "bot_noisy":
        # Correlated lateral disturbances create departures and gradual corrections.
        clearance = min(start[0], target[0], start[1], target[1],
                        width - 1 - start[0], width - 1 - target[0],
                        height - 1 - start[1], height - 1 - target[1])
        amplitude = min(12.0, distance * 0.025, clearance * 0.5)
        plan["noise_amplitude"] = amplitude
        plan["noise_knots"] = [0.0] + [rng.uniform(-amplitude, amplitude) for _ in range(7)] + [0.0]

    return plan


# Evaluate the same plan at any elapsed time; scheduling never changes its shape.
def bot_position(plan, elapsed):
    tau = min(1.0, max(0.0, elapsed / plan["duration"]))
    start, target = plan["start"], plan["target"]

    # 1. בוט Minimum Jerk: פולינום מעלה 5 (האצה ובלימה חלקה)
    if plan["source_type"] == "bot_smart_jerk":
        u = 10 * (tau**3) - 15 * (tau**4) + 6 * (tau**5)
        control = plan["control"]
        return tuple((1-u)**2 * start[i] + 2*(1-u)*u*control[i] + u*u*target[i] for i in (0, 1))

    # 2. בוט ביומכני: פרופיל מהירות א-סימטרי + רעידות נוירו-מוטוריות (10Hz)
    if plan["source_type"] == "bot_smart_full":
        tau_w = tau ** 0.8  # שיא מהירות מוקדם (ב-35-40% מהמסלול, כמו יד אנושית)
        u = 10 * (tau_w**3) - 15 * (tau_w**4) + 6 * (tau_w**5)
        u = min(1.0, max(0.0, u))
        control = plan["control"]
        base = tuple((1-u)**2 * start[i] + 2*(1-u)*u*control[i] + u*u*target[i] for i in (0, 1))
        # מעטפת רעידות שמתחילה ומסתיימת באפס
        env = 4.0 * u * (1.0 - u)
        phi1, phi2 = plan.get("tremor_phases", (0.0, 0.0))
        tremor = 2.5 * env * (0.7 * math.sin(2 * math.pi * 10.0 * elapsed + phi1) +
                              0.3 * math.cos(2 * math.pi * 18.0 * elapsed + phi2))
        return tuple(base[i] + tremor * plan["normal"][i] for i in (0, 1))

    # שאר הבוטים הנאיביים (מהירות קבועה u = tau)
    u = tau
    if plan["source_type"] == "bot_curved":
        control = plan["control"]
        return tuple((1-u)**2 * start[i] + 2*(1-u)*u*control[i] + u*u*target[i]
                     for i in (0, 1))
    base = tuple(start[i] + u*(target[i]-start[i]) for i in (0, 1))
    if plan["source_type"] == "bot_noisy":
        knots = plan["noise_knots"]
        position = u * (len(knots)-1)
        index = min(int(position), len(knots)-2)
        fraction = position - index
        # Smoothstep makes each correction continuous with a gradual onset/stop.
        weight = fraction*fraction*(3-2*fraction)
        offset = (1-weight)*knots[index] + weight*knots[index+1]
        return tuple(base[i] + offset*plan["normal"][i] for i in (0, 1))
    return base


def update_controls():
    # A running human or bot trial owns the sampler until completion/cancellation.
    for button in bot_buttons:
        button.config(state="disabled" if recording else "normal")
    cancel_button.config(state="normal" if recording else "disabled")


# A single entry point gives human and bot recordings identical metadata and timing.
def begin_trial(source_type, first_x, first_y, plan=None):
    global recording, trajectory, start_time, trial_metadata, active_source, bot_plan
    if recording:
        return
    active_source = source_type
    bot_plan = plan
    canvas.delete("trajectory")
    # Freeze all metadata so later layout changes cannot change a saved trial.
    trial_metadata = {
        "canvas_width": canvas.winfo_width(), "canvas_height": canvas.winfo_height(),
        "sample_interval_ms": SAMPLE_INTERVAL_MS, "start_x": start_x, "start_y": start_y,
        "target_x": target_x, "target_y": target_y, "source_type": source_type,
        "bot_seed": plan["seed"] if plan else "",
        "planned_duration_s": plan["duration"] if plan else "",
        "control_x": plan["control"][0] if plan and plan["control"] else "",
        "control_y": plan["control"][1] if plan and plan["control"] else "",
        "noise_amplitude_px": plan["noise_amplitude"] if plan else "",
        "generator_version": plan["generator_version"] if plan else "",
    }
    recording = True
    start_time = time.perf_counter()
    trajectory = [{"x": first_x, "y": first_y, "time": 0.0}]
    update_controls()
    status_label.config(text=("מקליט... הזיזי את העכבר אל העיגול האדום" if plan is None
                              else "הבוט מצייר מסלול... ההקלטה תסתיים בהגעה לאדום"))
    sample_mouse_position()


def run_bot(source_type):
    if recording:
        return
    seed = random.SystemRandom().randrange(2**32)
    plan = make_bot_plan(source_type, (start_x, start_y), (target_x, target_y),
                         canvas.winfo_width(), canvas.winfo_height(), seed)
    begin_trial(source_type, *plan["start"], plan=plan)


def run_linear_bot():
    run_bot("bot_linear")


def run_curved_bot():
    run_bot("bot_curved")


def run_noisy_bot():
    run_bot("bot_noisy")


def run_smart_jerk_bot():
    run_bot("bot_smart_jerk")


def run_smart_full_bot():
    run_bot("bot_smart_full")


# Start a fresh recording only when the user clicks inside the green circle.
def start_recording(event):
    # Ignore extra start clicks while a recording is already active.
    if recording:
        return

    if not inside_circle(
        event.x,
        event.y,
        start_x,
        start_y,
        radius
    ):
        return

    begin_trial("human", event.x, event.y)
    
# Append one timestamped position and check whether it reaches the target.
def add_sample(x, y, elapsed_time=None):
    if not recording:
        return

    # The previous position is used only to draw the next visible segment.
    previous = trajectory[-1]

    # Keep stationary samples so pauses retain their actual timestamps.
    if elapsed_time is None:
        elapsed_time = time.perf_counter() - start_time

    trajectory.append({
        "x": x,
        "y": y,
        "time": elapsed_time
    })

    # Draw only when the position changes.
    if x != previous["x"] or y != previous["y"]:
        canvas.create_line(
            previous["x"],
            previous["y"],
            x,
            y,
            fill=SOURCE_COLORS[active_source],
            width=2,
            tags="trajectory"
        )

    # Finish if the current point is inside the target, OR if the movement segment crossed it.
    if inside_circle(x, y, target_x, target_y, radius) or segment_intersects_circle(
        previous["x"], previous["y"], x, y, target_x, target_y, radius
    ):
        finish_recording()



# Poll human positions or evaluate the bot with the same real-time callback loop.
def sample_mouse_position():
    global sampling_job

    # The current sampling callback has already started
    sampling_job = None

    if not recording:
        return

    if active_source == "human":
        screen_x, screen_y = root.winfo_pointerxy()
        x = screen_x - canvas.winfo_rootx()
        y = screen_y - canvas.winfo_rooty()
        elapsed = time.perf_counter() - start_time
    else:
        elapsed = time.perf_counter() - start_time
        x, y = bot_position(bot_plan, elapsed)
        # Human pointer coordinates are integer pixels; use the same precision.
        # The underlying linear bot is straight; raster samples may form tiny steps.
        x, y = round(x), round(y)

    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()

    # Leaving the recorded area makes the trajectory incomplete.
    if not (0 <= x < canvas_width and 0 <= y < canvas_height):
        invalidate_trial("העכבר יצא ממשטח הציור")
        return

    add_sample(x, y, elapsed)

    # Reaching the target may have ended recording during add_sample.
    # Schedule another read only if the trial is still active.
    if recording:
        sampling_job = root.after(
            SAMPLE_INTERVAL_MS,
            sample_mouse_position
        )


# Stop an invalid trial without writing a CSV file.
def invalidate_trial(reason):
    global recording, trajectory, sampling_job, start_time

    if not recording:
        return

    recording = False
    # Cancel any pending read so it cannot continue after this trial ends.
    if sampling_job is not None:
        root.after_cancel(sampling_job)
        sampling_job = None

    # Retry with the same participant ID and circle positions after cancellation.
    trajectory = []
    start_time = None
    canvas.delete("trajectory")
    update_controls()
    status_label.config(
        text=f"הניסיון בוטל ולא נשמר: {reason}. אפשר לנסות שוב עם אותו מזהה"
    )



# Handle Tkinter leave events; the event argument is supplied by the binding.
def on_canvas_leave(event):
    if not recording or active_source != "human":
        return
    screen_x, screen_y = root.winfo_pointerxy()
    x = screen_x - canvas.winfo_rootx()
    y = screen_y - canvas.winfo_rooty()
    if not (0 <= x < canvas.winfo_width() and 0 <= y < canvas.winfo_height()):
        invalidate_trial("העכבר יצא ממשטח הציור")


# Handle real-time hardware mouse motion events to capture fast movements with zero delay.
def on_mouse_move(event):
    if not recording or active_source != "human":
        return

    # Check if inside canvas
    if not (0 <= event.x < canvas.winfo_width() and 0 <= event.y < canvas.winfo_height()):
        invalidate_trial("העכבר יצא ממשטח הציור")
        return

    # Record point only if it moved from the previous position
    if trajectory and (event.x != trajectory[-1]["x"] or event.y != trajectory[-1]["y"]):
        elapsed = time.perf_counter() - start_time
        add_sample(event.x, event.y, elapsed)
# Save the completed trial before advancing the ID and preparing the next layout.
def finish_recording():
    global recording, sampling_job, trajectory, start_time
    global participant_number, participant_id

    # Ignore stale completion callbacks after a trial has already ended.
    if not recording:
        return

    if len(trajectory) < 2:
        invalidate_trial("אין מספיק דגימות לשמירה")
        return

    recording = False
    if sampling_job is not None:
        root.after_cancel(sampling_job)
        sampling_job = None

    # Write with the completed trial's ID and circle positions before changing them.
    try:
        save_trajectory()
    except OSError as error:
        update_controls()
        status_label.config(text="השמירה נכשלה; המזהה לא התקדם. פרטי השגיאה מופיעים בחלון ההרצה")
        print(f"Save failed: {error}")
        return
    saved_participant = participant_id

    participant_number += 1
    participant_id = f"P{participant_number:02d}"
    participant_label.config(text=f"מזהה משתתף: {participant_id}")
    trajectory = []
    start_time = None
    canvas.delete("trajectory")
    randomize_circles()
    update_controls()

    # Preparing the next layout does not start recording; a green-circle click does.
    status_label.config(
        text=f"המסלול {saved_participant} נשמר! לחצי על הירוק או בחרי בוט לניסיון הבא"
    )
    print("Recording finished")


# Write one row per sample, repeating trial metadata to keep each row interpretable.
def save_trajectory():
    # Keep an internal random trajectory ID while using a simple visible filename.
    trajectory_id = uuid.uuid4().hex[:8]

    file_path = (
        SESSION_DIRECTORY /
        f"{participant_id}.csv"
    )

    # UTF-8 supports text metadata; newline="" lets csv handle line endings.
    with file_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        # Column order must match the values written for each sample below.
        # Positions and dimensions are pixels; time is seconds; interval is milliseconds.
        writer.writerow([
            "trajectory_id",
            "participant_id",
            "sample_index",
            "x",
            "y",
            "time",
            "start_x",
            "start_y",
            "target_x",
            "target_y",
            "canvas_width",
            "canvas_height",
            "sample_interval_ms",
            "session_id",
            "source_type",
            "bot_seed",
            "planned_duration_s",
            "control_x",
            "control_y",
            "noise_amplitude_px",
            "generator_version"
        ])

        # Sample indices begin at zero and follow the recording order.
        for index, point in enumerate(trajectory):
            writer.writerow([
                trajectory_id,
                participant_id,
                index,
                point["x"],
                point["y"],
                point["time"],
                trial_metadata["start_x"],
                trial_metadata["start_y"],
                trial_metadata["target_x"],
                trial_metadata["target_y"],
                trial_metadata["canvas_width"],
                trial_metadata["canvas_height"],
                trial_metadata["sample_interval_ms"],
                SESSION_ID,
                trial_metadata["source_type"],
                trial_metadata["bot_seed"],
                trial_metadata["planned_duration_s"],
                trial_metadata["control_x"],
                trial_metadata["control_y"],
                trial_metadata["noise_amplitude_px"],
                trial_metadata["generator_version"]
            ])

    print(f"Saved: {file_path}")


# Bot controls share the existing window and are disabled during every recording.
bot_frame = tk.Frame(root)
bot_frame.pack(pady=8)
bot_buttons = []
for label, command, color in (
    ("בוט ליניארי", run_linear_bot, "purple"),
    ("בוט מעוקל", run_curved_bot, "orange"),
    ("בוט רועש", run_noisy_bot, "teal"),
    ("בוט Min Jerk", run_smart_jerk_bot, "magenta"),
    ("בוט ביומכני", run_smart_full_bot, "brown"),
):
    button = tk.Button(bot_frame, text=label, command=command, fg=color)
    button.pack(side=tk.LEFT, padx=5)
    bot_buttons.append(button)
cancel_button = tk.Button(bot_frame, text="ביטול ניסיון", state="disabled",
                          command=lambda: invalidate_trial("הניסיון בוטל ידנית"))
cancel_button.pack(side=tk.LEFT, padx=5)


# Connect mouse events to their functions
# Connect mouse events to their functions
canvas.bind("<Button-1>", start_recording)
canvas.bind("<Leave>", on_canvas_leave)
canvas.bind("<Motion>", on_mouse_move)
canvas.bind("<B1-Motion>", on_mouse_move)

# Wait for layout before choosing centers, and keep the full drawing area visible.
root.update_idletasks()
root.minsize(root.winfo_reqwidth(), root.winfo_reqheight())
randomize_circles()
def on_close():
    global sampling_job
    if sampling_job is not None:
        root.after_cancel(sampling_job)
        sampling_job = None
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
# Run the event loop that processes clicks, leave events and scheduled samples.
root.mainloop()
