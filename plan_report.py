# Planning report for --plan option of every demo
#   * compare_methods: evidence for choosing jtraj or RMRC on a move. Plans
#     the same move both ways and measures
#   * print_plant: every move of robot task in order

import numpy as np

from motion_planner import DT


def compare_methods(planner, job, cases, title="JTRAJ vs RMRC on the same moves"):
    """
    Comapare two moves planned with jtraj and RMRC.
    Path error = worst distance of TCP from intended cartesian path
    Dip = how far TCP drops below intended path.
    Surface are supports (not obstacles) so collision check cannot see this.
    It is a property of the interpolation method itself.
    
    cases: (label, starting target, goal target, rmrc mode, holding) tuples. Targets
    are named as in the job's POSES
    """
    print("\n" + title)
    print("-" * 100)
    print(f"{'move':24s}{'method':7s}{'steps':>6s}{'path err':>10s}{'dip':>9s}{'max tilt':>11s}"
          f"{'peak v':>10s}{'min manip':>11s}   collision check")
    for label, a, b, mode, holding in cases:
        q_a = planner.solve_ik(job.target_pose(a), np.zeros(planner.robot.n), job.ik_seeds(a))
        q_b = planner.solve_ik(job.target_pose(b), q_a, job.ik_seeds(b))
        T_a = planner.tcp(q_a)
        x_ref, yaw_ref = planner.cartesian_path(T_a, job.target_polar(b), mode)
        # Dense reference path for the deviation measure
        x_dense = np.array([x_ref[:, i] for i in range(x_ref.shape[1])])

        q_j = planner.fine_jtraj(q_a, q_b)
        q_r, info = planner.rmrc(q_a, x_ref, yaw_ref)

        for method, qm in (("jtraj", q_j), ("RMRC", q_r)):
            tcps = [planner.tcp(q) for q in qm]
            dev = max(np.min(np.linalg.norm(x_dense - T.t, axis=1)) for T in tcps)
            # Height below the reference at the same fraction of the move
            idx = np.linspace(0, x_ref.shape[1] - 1, len(tcps)).round().astype(int)
            dip = max(0.0, max(x_ref[2, k] - T.t[2] for k, T in zip(idx, tcps)))
            tilt = max(planner.payload_tilt(T) for T in tcps)
            m = min(np.sqrt(abs(np.linalg.det(planner.robot.jacob0(q) @ planner.robot.jacob0(q).T)))
                    for q in qm[::5])
            pts = np.array([T.t for T in tcps])
            v_peak = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) / DT)
            hit = planner.check(qm, holding=holding)
            print(f"{label:24s}{method:7s}{len(qm):6d}{dev * 1000:8.1f}mm{dip * 1000:7.1f}mm"
                  f"{np.rad2deg(tilt):9.2f}deg{v_peak:7.3f}m/s{m:11.4f}   {hit or 'clear'}")
    print()


def print_plan(sequencer, lane, title):
    print(title)
    print("-" * 92)
    for name, method, steps, dur, note in sequencer.dry_run(lane):
        print(f"  {name:36s}{method:11s}{str(steps):>6s}{dur:>7s}   {note}")
    print()
