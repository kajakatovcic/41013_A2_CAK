# Shared task sequencer
# Runs every robot and the conveyor as separate lanes stepped together one
# frame at a time in a shared swift loop. Either for one station demo
# or the whole cell (main_scene.py)
#
# Each move is planned at the moment it starts from the robots actual state.
# This lets the cell stop betewen any two frames (e.g estop or light curtain)
# and resume by replanning the move from where it stopped. It also lets it
# see an obstacle that appears after the cycle started and replan around it
#
# Lanes are synchronised through shared flags. E.g. a segment with requires='x'
# waits until another segment with sets='x' has finished. E.g. robot waits for
# 'tray_at_jaka', conveyor waits for 'coffee_placed'
#
# A station must provide (in a "job" object) for its robot lane:
#   target_pose(name): world TCP pose of named target (jtraj)
#   target_polar(name): (r,phi,h) of a named target in the base frame (RMRC)
#   ik_seeds(name): extra IK seeds for a certain target
#   run_action(seg, lane) -> bool: one frame of a nonmotion segment (for tool 
#                    rleated tasks). Returns True when finished
#   update_visuals(lane): tool and held payload follow the flange
#
# A segment that picks up/releases something carries hold=True/False.
# The sequencer sets lane.holding when it finishes which siwtches the 
# payload's collision shapes on/off for planning

from math import ceil

import numpy as np
from roboticstoolbox import trapezoidal

from motion_planner import DT
from workcell_layout import BELT_SPEED

ROBOT_MOTIONS = ("jtraj", "rmrc")


class Segment:
    def __init__(self, kind, name, requires=None, sets=None, **params):
        self.kind = kind
        self.name = name
        self.requires = requires
        self.sets = sets
        self.p = params
        self.plan = None # joint matrix (or x values for the conveyor)
        self.i = 0 # next row to execute
        self.note = ""
        self.started = False


def method_label(seg):
    if seg.kind == "rmrc":
        return "RMRC " + seg.p.get("mode", "line")
    return seg.kind


class Lane:
    def __init__(self, name, segments):
        self.name = name
        self.segments = segments
        self.k = 0
        self.status = "waiting for START"

    @property
    def current(self):
        return self.segments[self.k] if self.k < len(self.segments) else None

    @property
    def done(self):
        return self.k >= len(self.segments)


class RobotLane(Lane):
    """The robot including its segment and station job"""

    def __init__(self, name, robot, planner, job, segments):
        super().__init__(name, segments)
        self.robot = robot
        self.planner = planner
        self.job = job
        self.holding = False

    def plan(self, seg, q_now, holding):
        """
        Plan one motion segment from the given state.
        Returns (plan, note).
        """
        goal = seg.p["goal"]
        if seg.kind == "jtraj":
            target = self.job.target_pose(goal) if isinstance(goal, str) else goal
            seeds = self.job.ik_seeds(goal) if isinstance(goal, str) else []
            return self.planner.plan_jtraj(q_now, target, holding, seeds)
        return self.planner.plan_rmrc(q_now, self.job.target_polar(goal),
                                      seg.p.get("mode", "line"), holding)


class ConveyorLane(Lane):
    """The tray moving on the conveyor belt between station stops"""

    def __init__(self, name, tray, segments):
        super().__init__(name, segments)
        self.tray = tray

    def plan(self, seg):
        """Trapezoidal profile to the next stop, with step count set
        so the belt's peak speed is BELT_SPEED"""
        x0, x1 = self.tray.x, seg.p["x"]
        steps = max(2, ceil(1.5 * abs(x1 - x0) / (BELT_SPEED * DT)))
        s = trapezoidal(0, 1, steps).q
        return x0 + s * (x1 - x0), "trapezoidal"


class TaskSequencer:
    def __init__(self, safety, lanes, log=print):
        self.safety = safety
        self.lanes = lanes
        self.log = log
        self.flags = set()
        self.world_version = 0
        self._seen_version = 0

    @property
    def robot_lanes(self):
        return [l for l in self.lanes if isinstance(l, RobotLane)]

    @property
    def status(self):
        """One-line status of every robot lane for the GUI label"""
        return " | ".join(f"{l.name}: {l.status}" for l in self.robot_lanes)

    def notify_world_changed(self):
        """Called when an obstacle is added, moved or removed"""
        self.world_version += 1

    # Per-frame execution
    def tick(self):
        """Advance every lane by one frame if the safety state allows motion"""
        if not self.safety.motion_allowed:
            return

        if self.safety.resumed:
            self.safety.resumed = False
            for lane in self.lanes:
                seg = lane.current
                if seg is not None and seg.plan is not None and \
                        (seg.kind in ROBOT_MOTIONS or seg.kind == "convey"):
                    seg.plan, seg.i = None, 0 # re-plan from where it halted

        if self.world_version != self._seen_version:
            self._seen_version = self.world_version
            for lane in self.robot_lanes:
                self._recheck(lane)
                if not self.safety.motion_allowed:
                    return

        for lane in self.lanes:
            self._tick_lane(lane)
            if not self.safety.motion_allowed:
                return

        if all(l.done for l in self.lanes):
            self.safety.complete()

    def _recheck(self, lane):
        """
        Collision prediction on remianing part of the move. If the new obstacle
        is in the way, replan from current joint state for active avoidance.
        The cell sotps if no safe route exists
        """
        seg = lane.current
        if seg is None or seg.kind not in ROBOT_MOTIONS or seg.plan is None:
            return
        remaining = seg.plan[seg.i:]
        if len(remaining) < 2:
            return
        reason = lane.planner.check(remaining, lane.holding)
        if reason is None:
            self.safety.note(f"{lane.name}: obstacle change checked, '{seg.name}' still clear.")
            return
        self.safety.note(f"{lane.name}: collision predicted on '{seg.name}' ({reason}). Re-planning...")
        plan, note = lane.plan(seg, lane.robot.q, lane.holding)
        if plan is None:
            self.safety.fault(f"{lane.name} '{seg.name}': {note}.")
            return
        seg.plan, seg.i, seg.note = plan, 0, note
        self.safety.note(f"{lane.name} '{seg.name}': {note}.")

    def _tick_lane(self, lane):
        seg = lane.current
        if seg is None:
            lane.status = "done"
            return
        if seg.requires and seg.requires not in self.flags:
            lane.status = f"waiting: {seg.requires.replace('_', ' ')}"
            return

        if not seg.started:
            seg.started = True
            self.log(f"[{lane.name}] {seg.name}")

        if seg.kind in ROBOT_MOTIONS or seg.kind == "convey":
            if seg.plan is None:
                if seg.kind == "convey":
                    plan, note = lane.plan(seg)
                else:
                    plan, note = lane.plan(seg, lane.robot.q, lane.holding)
                if plan is None:
                    self.safety.fault(f"{lane.name}: cannot plan '{seg.name}': {note}.")
                    return
                seg.plan, seg.i, seg.note = plan, 0, note
                if note not in ("direct", "trapezoidal"):
                    self.safety.note(f"{lane.name} '{seg.name}': {note}.")
            row = seg.plan[seg.i]
            if seg.kind == "convey":
                lane.tray.set_x(row)
            else:
                lane.robot.q = row
            seg.i += 1
            finished = seg.i >= len(seg.plan)
        else:
            finished = lane.job.run_action(seg, lane)

        lane.status = f"{seg.name} [{method_label(seg)}]"
        if finished:
            if "hold" in seg.p:
                lane.holding = seg.p["hold"]
            if seg.sets:
                self.flags.add(seg.sets)
            lane.k += 1

    def update_visuals(self):
        for lane in self.robot_lanes:
            lane.job.update_visuals(lane)

    # No swift run for plan_report
    def dry_run(self, lane):
        """
        Plan every move of a robot in order, each from the end of the previous
        move. Return rows for a summary table. Restores robot's joint state 
        afterwards
        """
        q0 = np.array(lane.robot.q, dtype=float)
        q = q0.copy()
        holding = False
        rows = []
        for seg in lane.segments:
            if "hold" in seg.p:
                holding = seg.p["hold"]
            if seg.kind not in ROBOT_MOTIONS:
                continue
            plan, note = lane.plan(seg, q, holding)
            if plan is None:
                rows.append((seg.name, method_label(seg), "-", "-", "FAILED: " + note))
                break
            rows.append((seg.name, method_label(seg), len(plan), f"{len(plan) * DT:.1f}s", note))
            q = plan[-1]
        lane.robot.q = q0
        return rows
