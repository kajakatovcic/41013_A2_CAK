from math import hypot, atan2

import numpy as np
from scipy.optimize import least_squares
from spatialmath import SE3

from lynxmotion_ses_pro_model import TOOL_Z
from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from workcell_layout import COLLECTION_X, CONVEYOR_Y, Z_BELT
from workcell_scene import Tray
from lynx_station import COOKIE_T

DEG = np.pi / 180

COOKIES_PER_ORDER = 1
PLATE_COOKIE_DX = [-0.045, 0.045]
PLATE_TOP = 0.015    
HOVER = 0.10         

FINAL_Q_DEG = None

Q_READY = np.array([135, -35, 75, 0, -40, 0]) * DEG

IK_SEEDS = [
    np.array([0, -35, 75, 0, -40, 0]) * DEG,
    np.array([0, -20, 50, 0, -30, 0]) * DEG,
    np.array([0, -55, 100, 0, -45, 0]) * DEG,
    np.array([0, -70, 130, 0, -60, 0]) * DEG,    
    np.array([0, -10, 30, 0, -20, 0]) * DEG,     
]

def tool_down_ik(robot, T_target):
    p = T_target.t
    yaw = float(np.arctan2(T_target.A[1, 0], T_target.A[0, 0]))
    lo, hi = robot.qlim[0], robot.qlim[1]

    def to_q(x):
        q6 = (yaw - x[0] + np.pi) % (2 * np.pi) - np.pi
        return np.array([x[0], x[1], x[2], 0.0, -(x[1] + x[2]), q6])

    def resid(x):
        return (robot.fkine(to_q(x)) * SE3.Tz(TOOL_Z)).t - p

    closest = np.inf
    for q3 in (1.2, 0.4, 2.0):                   
        for q2 in (-0.6, -1.2, 0.0, 0.6):
            for d1 in (0.0, 0.5, -0.5):
                sol = least_squares(resid, np.array([yaw + d1, q2, q3]),
                                    bounds=([-2 * np.pi, -np.pi, -np.pi], [2 * np.pi, np.pi, np.pi]))
                err = float(np.linalg.norm(sol.fun))
                q = to_q(sol.x)
                closest = min(closest, err)
                if err < 1e-3 and np.all(q >= lo) and np.all(q <= hi):
                    return q, err
    return None, closest

class LynxJob:
    def __init__(self, station, tray):
        self.station = station
        self.tray = tray
        self.held = None
        self.placed = 0

        self.tray_stop_x = station.base.t[0] - Tray.PLATE_SLOT.t[0]
        plate = SE3(self.tray_stop_x, CONVEYOR_Y, Z_BELT) * Tray.PLATE_SLOT

        self.targets = {}
        for k in range(COOKIES_PER_ORDER):
            pick = station.cookie_rest_pose(k).t
            place = (plate * SE3(PLATE_COOKIE_DX[k], 0, PLATE_TOP + COOKIE_T / 2)).t
            for name, p in ((f"pick_{k}", pick), (f"place_{k}", place)):
                r, phi, h = self._polar(p)
                self.targets[name] = (r, phi, h)
            for name, base_name in ((f"above_cookie_{k}", f"pick_{k}"), (f"above_plate_{k}", f"place_{k}")):
                r, phi, h = self.targets[base_name]
                self.targets[name] = (r, phi, h + HOVER)

        self.q_goal = {}
        for name, (r, phi, h) in self.targets.items():
            q, err = tool_down_ik(station.robot, self.target_pose(name))
            self.q_goal[name] = q
            if q is None:
                print(f"[Lynx] WARNING {name}: r={r:.3f} m, h={h:.3f} m is NOT reachable tool-down "
                      f"(closest approach {err * 1000:.0f} mm)")

    def _polar(self, p_world):
        p = (self.station.robot.base.inv() * SE3(*p_world)).t
        return float(hypot(p[0], p[1])), float(atan2(p[1], p[0])), float(p[2])

    def segments(self):
        S = Segment
        segs = [S("jtraj", "home -> ready", goal=Q_READY)]
        for k in range(COOKIES_PER_ORDER):
            n = k + 1
            segs += [
                S("jtraj", f"cookie {n}: move above cookie", goal=self._goal(f"above_cookie_{k}")),
                S("rmrc", f"cookie {n}: lower onto cookie", goal=f"pick_{k}", mode="line"),
                S("grip", f"cookie {n}: close gripper", close=True, hold=True, cookie=k),
                S("rmrc", f"cookie {n}: lift cookie", goal=f"above_cookie_{k}", mode="line"),
                S("rmrc", f"cookie {n}: carry level to plate", goal=f"above_plate_{k}", mode="arc",
                  requires="tray_at_lynx"),
                S("rmrc", f"cookie {n}: lower onto plate", goal=f"place_{k}", mode="line"),
                S("grip", f"cookie {n}: release", close=False, hold=False),
                S("rmrc", f"cookie {n}: lift away", goal=f"above_plate_{k}", mode="line"),
            ]
        segs.append(S("jtraj", "return to ready", goal=Q_READY, sets="food_placed"))
        if FINAL_Q_DEG is not None:
            segs.append(S("jtraj", "final joint state (supplied)", goal=np.deg2rad(FINAL_Q_DEG)))
        return segs

    def target_pose(self, name):
        r, phi, h = self.targets[name]
        return (self.station.robot.base * SE3(r * np.cos(phi), r * np.sin(phi), h)
                * SE3.Rz(phi))

    def target_polar(self, name):
        return self.targets[name]

    def ik_seeds(self, name):
        if self.q_goal.get(name) is not None:
            return [self.q_goal[name]]
        phi = self.targets[name][1]
        seeds = []
        for s in IK_SEEDS:
            s = s.copy()
            s[0] = phi
            seeds.append(s)
        return seeds

    def run_action(self, seg, lane):
        if seg.kind == "grip":
            seg.i += 1
            if seg.i < 10:                    
                return False
            if seg.p["close"]:
                self.held = self.station.cookies[seg.p["cookie"]]
            elif self.held is not None:
                self.tray.attach(self.held, self.tray.pose.inv() * self.held.pose)
                self.held = None
                self.placed += 1
            return True
        return True

    def update_visuals(self, lane):
        self.station.update_visuals(lane.robot.q)
        if self.held is not None:
            tcp = lane.robot.fkine(lane.robot.q) * SE3.Tz(TOOL_Z)     
            self.held.set_pose(SE3(*tcp.t))

    def _goal(self, name):
        q = self.q_goal.get(name)
        return name if q is None else q    

COMPARE_CASES = [
    ("lower onto cookie", "above_cookie_0", "pick_0", "line", False),
    ("carry cookie to plate", "above_cookie_0", "above_plate_0", "arc", True),
    ("lower onto plate", "above_plate_0", "place_0", "line", True),
]

class LynxTask(TaskSequencer):

    def __init__(self, robot, planner, station, tray, safety, log=print):
        self.tray = tray
        self.job = LynxJob(station, tray)
        S = Segment
        conveyor = [
            S("convey", "tray -> Lynxmotion station", x=self.job.tray_stop_x, sets="tray_at_lynx"),
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="food_placed",
              sets="delivered"),
        ]
        lanes = [RobotLane("Lynxmotion", robot, planner, self.job, self.job.segments()),
                 ConveyorLane("conveyor", tray, conveyor)]
        super().__init__(safety, lanes, log)

    def dry_run(self, lane=None):
        return super().dry_run(lane or self.lanes[0])