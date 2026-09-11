"""Test bot geometry and the shared recording lifecycle without opening a window.

The production functions are loaded from mouse_tracker.py via AST so tests never
create real sessions or start Tkinter. All CSV writes go to a temporary folder.
"""
import ast
import csv
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
import uuid

from analyze_trajectories import analyze_single_trajectory


class Widget:
    def __init__(self):
        self.options = {}
        self.items = []
    def config(self, **kwargs):
        self.options.update(kwargs)
    def delete(self, tag):
        self.items = [item for item in self.items if item[1].get("tags") != tag]
    def create_line(self, *args, **kwargs):
        self.items.append((args, kwargs))
    create_oval = create_line
    def winfo_width(self):
        return 906
    def winfo_height(self):
        return 606
    def winfo_rootx(self):
        return 0
    def winfo_rooty(self):
        return 0


class BotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.clock = 10.0
        self.jobs = {}
        self.counter = 0
        self.pointer = (-100, -100)
        self.env = dict(csv=csv, math=math, random=random.Random(13), uuid=uuid,
            radius=25, CIRCLE_MARGIN=20, MIN_CENTER_DISTANCE=250, BOT_DURATION_S=1.0,
            SOURCE_COLORS={"human":"blue", "bot_linear":"purple", "bot_curved":"orange", "bot_noisy":"teal"},
            start_x=100, start_y=300, target_x=800, target_y=300,
            recording=False, trajectory=[], start_time=None, sampling_job=None,
            trial_metadata={}, active_source="human", bot_plan=None,
            participant_id="P01", participant_number=1, SAMPLE_INTERVAL_MS=5,
            SESSION_ID="session_test", SESSION_DIRECTORY=Path(self.directory.name),
            canvas=Widget(), status_label=Widget(), participant_label=Widget(),
            bot_buttons=[Widget(), Widget(), Widget()], cancel_button=Widget(),
            time=SimpleNamespace(perf_counter=self.now),
            root=SimpleNamespace(winfo_pointerxy=lambda:self.pointer,
                                 after=self.schedule, after_cancel=lambda key:self.jobs.pop(key, None)))
        # SystemRandom is only used to choose the saved seed; deterministic plans
        # can also be evaluated directly in the geometry tests below.
        self.env["random"] = random
        tree = ast.parse(Path(__file__).with_name("mouse_tracker.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<tracker functions>", "exec"), self.env)

    def now(self):
        self.clock += 0.0001
        return self.clock

    def schedule(self, delay, callback):
        self.assertEqual(delay, 5)
        self.counter += 1
        self.jobs[self.counter] = callback
        return self.counter

    def tick(self):
        self.clock += 0.007  # Real callbacks need not arrive exactly at the requested delay.
        key = next(iter(self.jobs))
        self.jobs.pop(key)()

    def finish(self):
        for _ in range(500):
            if not self.jobs:
                break
            self.tick()
        self.assertFalse(self.env["recording"])
        self.assertFalse(self.jobs)

    def test_geometry_bounds_seeds_and_orientations(self):
        self.env["random"] = random
        for start, target in [((100,300),(800,300)), ((450,60),(450,540)),
                              ((60,60),(845,545)), ((845,545),(60,60))]:
            for kind in ("bot_linear", "bot_curved", "bot_noisy"):
                for seed in range(30):
                    plan = self.env["make_bot_plan"](kind,start,target,906,606,seed)
                    self.assertEqual(plan,self.env["make_bot_plan"](kind,start,target,906,606,seed))
                    self.assertTrue(self.env["inside_circle"](*plan["start"],*start,25))
                    self.assertEqual(self.env["bot_position"](plan,0),plan["start"])
                    self.assertEqual(self.env["bot_position"](plan,1),target)
                    for i in range(101):
                        x,y=self.env["bot_position"](plan,i/100)
                        self.assertTrue(0 <= x < 906 and 0 <= y < 606)
                    if kind == "bot_curved":
                        sx,sy=plan["start"]; tx,ty=target; cx,cy=plan["control"]
                        self.assertGreater(abs((tx-sx)*(cy-sy)-(ty-sy)*(cx-sx)),1)

    def test_linear_speed_and_noise_departures(self):
        plan=self.env["make_bot_plan"]("bot_linear",(100,300),(800,300),906,606,8)
        positions=[self.env["bot_position"](plan,i/10) for i in range(11)]
        steps=[math.dist(a,b) for a,b in zip(positions,positions[1:])]
        self.assertLess(max(steps)-min(steps),1e-9)
        noisy=self.env["make_bot_plan"]("bot_noisy",(100,300),(800,300),906,606,8)
        self.assertGreater(max(math.dist(self.env["bot_position"](plan,i/20),
                                         self.env["bot_position"](noisy,i/20)) for i in range(21)),1)

    def test_each_bot_saves_same_schema_and_first_target_entry(self):
        for kind in ("bot_linear","bot_curved","bot_noisy"):
            identifier=self.env["participant_id"]
            self.env["run_bot"](kind)
            metadata=dict(self.env["trial_metadata"])
            self.assertTrue(all(b.options["state"]=="disabled" for b in self.env["bot_buttons"]))
            # The physical pointer is outside; it must not cancel a simulated bot.
            self.env["on_canvas_leave"](None)
            self.assertTrue(self.env["recording"])
            self.finish()
            path=Path(self.directory.name)/f"{identifier}.csv"
            with path.open() as stream: rows=list(csv.DictReader(stream))
            self.assertGreater(len(rows),2)
            self.assertTrue(all(row["source_type"]==kind and row["session_id"]=="session_test" for row in rows))
            self.assertEqual(float(rows[0]["target_x"]),metadata["target_x"])
            times=[float(row["time"]) for row in rows]
            self.assertTrue(all(b>a for a,b in zip(times,times[1:])))
            inside=[self.env["inside_circle"](float(row["x"]),float(row["y"]),
                    metadata["target_x"],metadata["target_y"],25) for row in rows]
            self.assertFalse(any(inside[:-1])); self.assertTrue(inside[-1])
            self.assertEqual(float(rows[0]["sample_interval_ms"]),5)
            self.assertEqual(analyze_single_trajectory(path)["source_type"],kind)
            self.assertTrue(all(b.options["state"]=="normal" for b in self.env["bot_buttons"]))

    def test_active_recording_cannot_be_replaced(self):
        self.env["run_bot"]("bot_linear")
        before=(self.env["active_source"],self.env["start_time"],id(self.env["trajectory"]))
        self.env["run_bot"]("bot_noisy")
        self.env["start_recording"](SimpleNamespace(x=100,y=300))
        self.assertEqual(before,(self.env["active_source"],self.env["start_time"],id(self.env["trajectory"])))
        self.env["invalidate_trial"]("test cancellation")
        self.assertEqual(self.env["participant_id"],"P01")
        self.assertFalse(self.jobs)
        self.assertFalse(list(Path(self.directory.name).glob('*.csv')))

    def test_human_after_bot_clears_bot_metadata_and_still_cancels(self):
        self.env["run_bot"]("bot_curved"); self.finish()
        self.pointer=(self.env["start_x"],self.env["start_y"])
        self.env["start_recording"](SimpleNamespace(x=self.pointer[0],y=self.pointer[1]))
        self.assertEqual(self.env["active_source"],"human")
        self.assertEqual(self.env["trial_metadata"]["bot_seed"],"")
        self.env["run_bot"]("bot_noisy")
        self.assertEqual(self.env["active_source"],"human")
        self.pointer=(self.env["target_x"],self.env["target_y"])
        self.tick()
        with (Path(self.directory.name)/'P02.csv').open() as f:rows=list(csv.DictReader(f))
        self.assertEqual(rows[0]["source_type"],"human")
        self.assertEqual(rows[0]["planned_duration_s"],"")
        self.pointer=(self.env["start_x"],self.env["start_y"])
        self.env["start_recording"](SimpleNamespace(x=self.pointer[0],y=self.pointer[1]))
        self.pointer=(-1,-1); self.env["on_canvas_leave"](None)
        self.assertFalse(self.env["recording"])
        self.assertEqual(self.env["participant_id"],"P03")


if __name__ == "__main__":
    unittest.main()
