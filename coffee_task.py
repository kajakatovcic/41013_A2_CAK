# The JAKA coffee order run as two lanes stepped together one frame at a time.
#
#   robot: ready -> pick cup -> brew -> carry -> place on tray -> ready
#   conveyor: bring the tray to the JAKA -> (wait for coffee) -> collection
#
# Pseudo-simultaneous via one shared swift loop but each move is planned
# at the moment it starts from the robot's actual state.
# It llows the cell to:
#   * halt between any two frames (e.g. e-stop, light curtain), and resume by
#     re-planning the interrupted move from where the arm stopped
#   * see an obstacle that appears after the cycle started (the milk pitcher)
#     and re-plan around it before reaching it.
#
# The lanes are synchronised via flags - the carry to the tray waits for
# 'tray_at_jaka', the conveyor waits for 'coffee_placed'.

from math import ceil

import numpy as np
from spatialmath import SE3
from roboticstoolbox import trapezoidal

from jaka_motion import DT, TOOL
from coffee_station import POSES, IK_SEEDS, Gripper
from workcell_layout import BELT_SPEED, COLLECTION_X
from workcell_scene import Tray

MOTION_KINDS = ("jtraj", "rmrc", "convey")


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

    @property
    def current(self):
        return self.segments[self.k] if self.k < len(self.segments) else None

    @property
    def done(self):
        return self.k >= len(self.segments)


class CoffeeTask:
    def __init__(self, robot, planner, station, tray, safety, final_q=None, log=print):
        self.robot = robot
        self.planner = planner
        self.station = station
        self.tray = tray
        self.safety = safety
        self.log = log
        self.flags = set()
        self.holding = False
        self.cup_on_tray = False
        self.world_version = 0
        self._seen_version = 0
        self.status = "waiting for START"

        # Conveyor stop for the JAKA- the tray halts with its CUP_SLOT exactly
        # at the planner's 'on_tray' target (r, phi) in the JAKA base frame.
        r, phi, _ = POSES["on_tray"]
        slot_x = (station.base * SE3(r * np.cos(phi), r * np.sin(phi), 0)).t[0]
        self.tray_stop_x = slot_x - Tray.CUP_SLOT.t[0]

        S = Segment
        robot_segments = [
            S("jtraj", "home -> ready", goal="ready"),
            S("event", "dispense a cup", action="dispense"),
            S("jtraj", "ready -> in front of cup", goal="pre_pick"),
            S("rmrc", "approach cup (straight line)", goal="pick", mode="line"),
            S("grip", "close gripper", close=True),
            S("rmrc", "lift cup off pad", goal="lift", mode="line"),
            S("jtraj", "empty cup -> coffee machine", goal="pre_insert"),
            S("rmrc", "insert cup into bay", goal="insert", mode="line"),
            S("brew", "brew coffee", frames=int(3.0 / DT)),
            S("rmrc", "withdraw full cup", goal="pre_insert", mode="line"),
            S("rmrc", "carry coffee to tray (level arc)", goal="above_tray", mode="arc",
              requires="tray_at_jaka"),
            S("rmrc", "lower cup onto tray", goal="on_tray", mode="line"),
            S("grip", "release cup", close=False),
            S("rmrc", "slide fingers off cup", goal="retreat", mode="line", sets="coffee_placed"),
            S("jtraj", "return to ready", goal="ready"),
        ]
        if final_q is not None:
            robot_segments.append(S("jtraj", "final joint state (supplied)", goal=np.asarray(final_q)))

        conveyor_segments = [
            S("convey", "tray -> JAKA station", x=self.tray_stop_x, sets="tray_at_jaka"),
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="coffee_placed",
              sets="delivered"),
        ]
        self.lanes = [Lane("JAKA", robot_segments), Lane("conveyor", conveyor_segments)]

    # Planning
    def _goal(self, seg):
        g = seg.p["goal"]
        return self.station.tcp_pose(g) if isinstance(g, str) else g

    def _seeds(self, seg):
        g = seg.p["goal"]
        if not isinstance(g, str):
            return []
        phi = POSES[g][1]
        seeds = []
        for s in IK_SEEDS:
            s = s.copy()
            s[0] = phi
            seeds.append(s)
        return seeds

    def plan_segment(self, seg, q_now, holding):
        """Plan one segment from the given state. Returns (plan, note)."""
        if seg.kind == "jtraj":
            return self.planner.plan_jtraj(q_now, self._goal(seg), holding, self._seeds(seg))
        if seg.kind == "rmrc":
            return self.planner.plan_rmrc(q_now, POSES[seg.p["goal"]], seg.p["mode"], holding)
        if seg.kind == "convey":
            x0, x1 = self.tray.x, seg.p["x"]
            steps = max(2, ceil(1.5 * abs(x1 - x0) / (BELT_SPEED * DT)))
            s = trapezoidal(0, 1, steps).q
            return x0 + s * (x1 - x0), "trapezoidal"
        return None, ""

    # Per-frame execution
    def notify_world_changed(self):
        """Call when an obstacle is added, moved or removed."""
        self.world_version += 1

    def tick(self):
        """Advance every lane by one frame, if the safety state allows motion."""
        if not self.safety.motion_allowed:
            return

        if self.safety.resumed:
            self.safety.resumed = False
            for lane in self.lanes:
                seg = lane.current
                if seg is not None and seg.kind in MOTION_KINDS and seg.plan is not None:
                    seg.plan, seg.i = None, 0 # re-plan from where it halted

        if self.world_version != self._seen_version:
            self._seen_version = self.world_version
            self._recheck_robot_lane()
            if not self.safety.motion_allowed:
                return

        for lane in self.lanes:
            self._tick_lane(lane)
            if not self.safety.motion_allowed:
                return

        if all(l.done for l in self.lanes):
            self.status = "order complete"
            self.safety.complete()

    def _recheck_robot_lane(self):
        """
        Collision prediction on the remaining part of the move in progress.
        If the new obstacle is in the way, re-plan from the current joint state
        (active avoidance). Only if no safe route exists does the cell stop.
        """
        seg = self.lanes[0].current
        if seg is None or seg.kind not in ("jtraj", "rmrc") or seg.plan is None:
            return
        remaining = seg.plan[seg.i:]
        if len(remaining) < 2:
            return
        reason = self.planner.check(remaining, self.holding)
        if reason is None:
            self.safety.note(f"Obstacle change checked: '{seg.name}' still clear.")
            return
        self.safety.note(f"Collision predicted on '{seg.name}' ({reason}). Re-planning...")
        plan, note = self.plan_segment(seg, self.robot.q, self.holding)
        if plan is None:
            self.safety.fault(f"'{seg.name}': {note}.")
            return
        seg.plan, seg.i, seg.note = plan, 0, note
        self.safety.note(f"'{seg.name}': {note}.")

    def _tick_lane(self, lane):
        seg = lane.current
        if seg is None:
            return
        if seg.requires and seg.requires not in self.flags:
            if lane.name == "JAKA":
                self.status = f"waiting: {seg.requires.replace('_', ' ')}"
            return

        if not seg.started:
            seg.started = True
            self.log(f"[{lane.name}] {seg.name}")

        if seg.kind in MOTION_KINDS:
            if seg.plan is None:
                q_now = self.robot.q
                plan, note = self.plan_segment(seg, q_now, self.holding)
                if plan is None:
                    self.safety.fault(f"Cannot plan '{seg.name}': {note}.")
                    return
                seg.plan, seg.i, seg.note = plan, 0, note
                if note not in ("direct", "trapezoidal"):
                    self.safety.note(f"'{seg.name}': {note}.")
            row = seg.plan[seg.i]
            if seg.kind == "convey":
                self.tray.set_x(row)
            else:
                self.robot.q = row
            seg.i += 1
            finished = seg.i >= len(seg.plan)

        elif seg.kind == "grip":
            target = Gripper.CLOSED if seg.p["close"] else Gripper.OPEN
            g = self.station.gripper
            step = (Gripper.OPEN - Gripper.CLOSED) / 10
            g.opening = target if abs(g.opening - target) <= step else g.opening + np.sign(target - g.opening) * step
            finished = g.opening == target
            if finished:
                self._grip_done(seg.p["close"])

        elif seg.kind == "brew":
            if seg.i == 0:
                self.station.set_brewing(True)
            seg.i += 1
            finished = seg.i >= seg.p["frames"]
            if finished:
                self.station.set_brewing(False, done=True)

        elif seg.kind == "event":
            if seg.p["action"] == "dispense":
                self.station.cup.set_pose(self.station.cup_dispense_pose)
            finished = True
        else:
            finished = True

        if lane.name == "JAKA":
            self.status = f"{seg.name} [{method_label(seg)}]"
        if finished:
            if seg.sets:
                self.flags.add(seg.sets)
            lane.k += 1

    def _grip_done(self, closed):
        if closed:
            self.holding = True
        else:
            self.holding = False
            # Hand the cup to the tray at its exact current pose.
            cup = self.station.cup
            self.tray.attach(cup, self.tray.pose.inv() * cup.pose)
            self.cup_on_tray = True

    def update_visuals(self):
        """Gripper and held cup follow the flange every frame."""
        T_flange = self.robot.fkine(self.robot.q)
        self.station.gripper.update(T_flange)
        if self.holding:
            self.station.cup.held_by(T_flange * TOOL)


    #dry run of the robot lane (no Swift) for --plan and general preplanning
    def dry_run(self):
        """
        Plan every robot move in order, each from the end of the previous one,
        and return rows for a summary table. Robot state is restored afterwards.
        """
        q0 = np.array(self.robot.q, dtype=float)
        q = q0.copy()
        holding = False
        rows = []
        for seg in self.lanes[0].segments:
            if seg.kind == "grip":
                holding = seg.p["close"]
                continue
            if seg.kind not in ("jtraj", "rmrc"):
                continue
            plan, note = self.plan_segment(seg, q, holding)
            if plan is None:
                rows.append((seg.name, method_label(seg), "-", "-", "FAILED: " + note))
                break
            rows.append((seg.name, method_label(seg), len(plan), f"{len(plan) * DT:.1f}s", note))
            q = plan[-1]
        self.robot.q = q0
        return rows
