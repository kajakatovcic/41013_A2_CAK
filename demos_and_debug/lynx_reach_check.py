import os
import sys
# This script lives in demos_and_debug/: put the repo root (one level up) on
# the import path so the shared workcell modules are found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from spatialmath import SE3

import lynxmotion_ses_pro_model as M
from lynx_station import build_lynx
from lynx_task import LynxJob, IK_SEEDS, Q_READY

DEG = np.pi / 180

def try_ik(robot, planner, T, seeds):
    """'#' reachable within joint limits, 'L' reachable only outside them,
    '.' unreachable. T is a world TCP pose; like the shared planner, it is
    converted to the flange with the inverse tool transform."""
    T = T * planner.settings.tool.inv()
    result = "."
    lo, hi = robot.qlim[0], robot.qlim[1]
    for q0 in seeds:
        try:
            sol = robot.ikine_LM(T, q0=q0, ilimit=60, slimit=5, seed=0, joint_limits=False)
        except TypeError:
            sol = robot.ikine_LM(T, q0=q0, ilimit=60, slimit=5, seed=0)
        if not sol.success:
            continue
        if np.all(sol.q >= lo) and np.all(sol.q <= hi):
            return "#"
        result = "L"
    return result

def seeds_for(phi):
    out = []
    for s in IK_SEEDS:
        for dphi in (0.0, 15 * DEG, -15 * DEG):
            s2 = s.copy()
            s2[0] = phi + dphi
            out.append(s2)
    return out

def main():
    robot, station, planner = build_lynx()

    print("ARM DIMENSIONS (m)")
    for name in ("D1", "Y_UP", "L_UPPER", "Y_FORE", "L_J5", "J6_OFFSET_Y", "TOOL_Z", "BASE_Z"):
        print(f"  {name:12s}{getattr(M, name):8.3f}")
    lateral = M.Y_UP + M.Y_FORE + M.J6_OFFSET_Y
    print(f"  sideways offset of the fingertips from the arm plane ~ {lateral:.3f}")

    T = robot.fkine(Q_READY)
    p = (robot.base.inv() * T).t
    print(f"\nREADY POSE fingertips: r = {np.hypot(p[0], p[1]):.3f}  h = {p[2]:.3f}  "
          f"(bearing {np.degrees(np.arctan2(p[1], p[0])):.0f} deg)")

    job = LynxJob(station, None)
    print("\nJOB TARGETS (r, bearing, h) in the robot base frame, and whether the arm reaches them")
    for name, (r, phi, h) in job.targets.items():
        flag = try_ik(robot, planner, job.target_pose(name), job.ik_seeds(name))
        print(f"  {name:16s} r={r:.3f}  phi={np.degrees(phi):7.1f} deg  h={h:7.3f}   {flag}")

    phi = 135 * DEG
    rs = np.arange(0.25, 0.86, 0.05)
    hs = [-0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    seeds = seeds_for(phi)
    print("\nREACH MAP, tool pointing down (columns = r in m, rows = h in m)")
    print("  h \\ r  " + " ".join(f"{r:4.2f}" for r in rs))
    for h in hs:
        row = []
        for r in rs:
            Tt = robot.base * SE3(r * np.cos(phi), r * np.sin(phi), h) * SE3.Rz(phi)
            row.append(f"  {try_ik(robot, planner, Tt, seeds)}  ")
        print(f"  {h:5.2f}  " + "".join(row))

if __name__ == "__main__":
    main()