"""
Explains why 'lower onto cookie' fails (no Swift needed).

    python demos_and_debug/lynx_debug.py

Prints, for the hover pose and the pick pose: the joint angles, the fingertip height and the height
of every link frame above the counter. Then it plans the straight-down move the way the planner does
and reports the real reason each attempt is rejected, plus a check that the Jacobian matches the
forward kinematics (they disagree if the model's built-in tool is not handled the same way in both).
"""

import os
import sys
# This script lives in demos_and_debug/: put the repo root (one level up) on
# the import path so the shared workcell modules are found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from lynx_station import build_lynx, MIN_LINK_Z
from lynx_task import LynxJob
from workcell_layout import Z_COUNTER
from motion_planner import HOP_HEIGHTS, RMRC_POS_TOL, RMRC_TILT_TOL

robot, station, planner = build_lynx()
job = LynxJob(station, None)
names = ["base"] + [l.name for l in robot.links]


def q_for(name):
    """Joint angles for a named target, solved the way the planner does
    (ikine_LM, joint limits, the job's seeds)."""
    return planner.solve_ik(job.target_pose(name), np.zeros(robot.n), job.ik_seeds(name))


def heights(q):
    """Height of every frame fkine_all returns, relative to the counter top (m)."""
    return [float(f.t[2]) - Z_COUNTER for f in list(robot.fkine_all(q))]


print(f"Planner requires every frame above counter + {MIN_LINK_Z - Z_COUNTER:.3f} m (except the base)\n")

for name in ("above_cookie_0", "pick_0"):
    q = q_for(name)
    print(f"== {name}  target (r, phi, h) = {tuple(round(v, 3) for v in job.targets[name])}")
    if q is None:
        print("   no tool-down solution found\n")
        continue
    print("   q (deg):", np.round(np.degrees(q), 1))
    print(f"   fingertip height above counter: {planner.tcp(q).t[2] - Z_COUNTER:.3f} m")
    hs = heights(q)
    for i, h in enumerate(hs):
        label = names[i] if len(hs) == len(names) else f"frame {i}"
        flag = "   <-- below limit" if i > 0 and h < MIN_LINK_Z - Z_COUNTER else ""
        print(f"   {label:14s} {h:7.3f} m{flag}")
    print()

q0 = q_for("above_cookie_0")
if q0 is not None:
    print("== JACOBIAN vs FORWARD KINEMATICS at the hover pose")
    J = robot.jacob0(q0)[:3]
    Jfd = np.zeros((3, 6))
    p0 = robot.fkine(q0).t
    for i in range(6):
        dq = q0.copy()
        dq[i] += 1e-6
        Jfd[:, i] = (robot.fkine(dq).t - p0) / 1e-6
    diff = float(np.abs(J - Jfd).max())
    print(f"   max |J - finite difference| = {diff:.5f}  ->", "OK" if diff < 1e-4 else "MISMATCH")

    print("\n== PLANNING 'lower onto cookie' (RMRC straight down), one line per attempt")
    T_start = planner.tcp(q0)
    for hop in (0.0,) + HOP_HEIGHTS:
        x, yaw = planner.cartesian_path(T_start, job.target_polar("pick_0"), "line", hop)
        qm, info = planner.rmrc(q0, x, yaw)
        tracked = info["pos_err"] <= RMRC_POS_TOL and info["max_tilt"] <= RMRC_TILT_TOL
        reason = planner.check(qm, False)
        lows = [min(heights(q)[1:]) for q in qm[::3]]
        print(f"   hop {hop:.2f} m: end error {info['pos_err'] * 1000:5.1f} mm, tilt {np.degrees(info['max_tilt']):4.1f} deg, "
              f"lowest frame {min(lows):6.3f} m, tracked={tracked}, check={reason or 'clear'}")