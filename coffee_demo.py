"""
Smooth Oper-Caterer: JAKA MiniCobo coffee station in the shared workcell.

    python coffee_demo.py                       # Swift demo, press START in the panel
    python coffee_demo.py --autostart           # start the cycle immediately
    python coffee_demo.py --final-q "90,-60,-90,0,60,0" # supply a final joint state
    python coffee_demo.py --plan                # no Swift: plan every move + print
                                                # the jtraj vs RMRC comparison

Swift panel controls (the simulated GUI e-stop and safety inputs):
    E-STOP / Release E-stop / RESET / START-RESUME
    Hand into light curtain   - asynchronous unsafe-zone signal at the exit
    Milk pitcher in path      - deliberately place an object in the carry path

Sequence: the tray (already holding the plate, food and drink from the
upstream stations) rides to the JAKA. Meanwhile the JAKA takes a cup from the
dispenser, brews a coffee, carries it level to the tray, and the conveyor takes
the finished order out through the light curtain to the collection zone.
"""

import argparse

import numpy as np
from spatialmath import SE3

from jakaminicobo import JakaMiniCobo
from jaka_motion import JakaPlanner, DT, TOOL, cup_tilt
from coffee_station import CoffeeStation, POSES, IK_SEEDS
from coffee_task import CoffeeTask
from safety_controller import SafetyController, LightCurtain
from workcell_scene import Workcell
from workcell_layout import (STATIONS, Z_COUNTER, LIGHT_CURTAIN_X, EXIT_HALF_WIDTH,
                             LIGHT_CURTAIN_Z)

# Link frames must stay 3cm above the counter - the wrist housing radius.
MIN_LINK_Z = Z_COUNTER + 0.03


def build(final_q=None, log=print):
    base = STATIONS["JAKA"][0]
    robot = JakaMiniCobo(base=base)
    station = CoffeeStation(base)
    cell = Workcell()
    planner = JakaPlanner(robot, station.obstacles, MIN_LINK_Z)
    planner.cup_height = 0.115
    curtain = LightCurtain(LIGHT_CURTAIN_X, EXIT_HALF_WIDTH - 0.06, LIGHT_CURTAIN_Z)
    return robot, station, cell, planner, curtain


def ik_at(planner, station, name, q_hint):
    seeds = []
    for s in IK_SEEDS:
        s = s.copy()
        s[0] = POSES[name][1]
        seeds.append(s)
    return planner.solve_ik(station.tcp_pose(name), q_hint, seeds)


def compare_methods(planner, station):
    """
    Compare different methods: Two moves planend with jtraj (IK at both ends,
    quintic joint interpolation) and with RMRC. 
    Path error = worst distane of TCP from intended Cartesian path
    Dip = how far the TCP drops below the intended path. On the slides along a
    surface (e.g cup placed into the bay) the cup/gripper fingers start level
    with the surface, so any dip drives them into it.
    Surfaces are supports (not obstacles) so the collision check cannot see this.
    It is a property of the interpolation method itself.
    """
    cases = [
        ("approach cup on pad", "pre_pick", "pick", "line"),
        ("insert cup into bay", "pre_insert", "insert", "line"),
        ("carry full cup to tray", "pre_insert", "above_tray", "arc"),
        ("slide fingers off cup", "on_tray", "retreat", "line"),
    ]
    print("\nJTRAJ vs RMRC on the same moves")
    print("-" * 100)
    print(f"{'move':24s}{'method':7s}{'steps':>6s}{'path err':>10s}{'dip':>9s}{'max tilt':>11s}"
          f"{'peak v':>10s}{'min manip':>11s}   collision check")
    for label, a, b, mode in cases:
        q_a = ik_at(planner, station, a, np.zeros(6))
        q_b = ik_at(planner, station, b, q_a)
        T_a = planner.tcp(q_a)
        x_ref, yaw_ref = planner.cartesian_path(T_a, POSES[b], mode)
        # Dense reference path for the deviation measure.
        x_dense = np.array([x_ref[:, i] for i in range(x_ref.shape[1])])

        q_j = planner.fine_jtraj(q_a, q_b)
        q_r, info = planner.rmrc(q_a, x_ref, yaw_ref)

        holding = a != "pre_pick"
        for method, qm in (("jtraj", q_j), ("RMRC", q_r)):
            tcps = [planner.tcp(q) for q in qm]
            dev = max(np.min(np.linalg.norm(x_dense - T.t, axis=1)) for T in tcps)
            # Height below the reference at the same fraction of the move
            idx = np.linspace(0, x_ref.shape[1] - 1, len(tcps)).round().astype(int)
            dip = max(0.0, max(x_ref[2, k] - T.t[2] for k, T in zip(idx, tcps)))
            tilt = max(cup_tilt(T) for T in tcps)
            m = min(np.sqrt(abs(np.linalg.det(planner.robot.jacob0(q) @ planner.robot.jacob0(q).T)))
                    for q in qm[::5])
            pts = np.array([T.t for T in tcps])
            v_peak = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) / DT)
            hit = planner.check(qm, holding=holding)
            print(f"{label:24s}{method:7s}{len(qm):6d}{dev * 1000:8.1f}mm{dip * 1000:7.1f}mm"
                  f"{np.rad2deg(tilt):9.2f}deg{v_peak:7.3f}m/s{m:11.4f}   {hit or 'clear'}")
    print()


def print_plan(task):
    print("PLANNED JAKA SEQUENCE (each move planned from the end of the previous)")
    print("-" * 92)
    for name, method, steps, dur, note in task.dry_run():
        print(f"  {name:36s}{method:11s}{str(steps):>6s}{dur:>7s}   {note}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="plan and report only, no Swift")
    parser.add_argument("--autostart", action="store_true", help="start without pressing START")
    parser.add_argument("--final-q", type=str, default=None,
                        help="final joint state in degrees, comma separated (6 values)")
    args = parser.parse_args()

    final_q = None
    if args.final_q:
        final_q = np.deg2rad([float(v) for v in args.final_q.split(",")])
        if len(final_q) != 6:
            parser.error("--final-q needs 6 joint values")

    robot, station, cell, planner, curtain = build()
    robot.q = np.zeros(6)

    if args.plan:
        safety = SafetyController()
        task = CoffeeTask(robot, planner, station, cell.conveyor.tray, safety, final_q)
        compare_methods(planner, station)
        print_plan(task)
        return

    import swift
    env = swift.Swift()
    env.launch(realtime=True)
    cell.add_to_env(env)
    station.add_to_env(env)
    robot.add_to_env(env)
    env.set_camera_pose([3.4, 2.6, 2.3], [1.4, -0.5, 0.9])

    state_label = swift.Label("STATE: IDLE")
    task_label = swift.Label("Task: waiting for START")
    safety = SafetyController(beacon=cell.beacon, curtain_field=cell.curtain_field,
                              label=state_label)
    task = CoffeeTask(robot, planner, station, cell.conveyor.tray, safety, final_q)

    def toggle_hand(_=None):
        inside = cell.toggle_hand()
        safety.note("Hand placed in the light curtain." if inside else "Hand removed from the light curtain.")

    def toggle_pitcher(_=None):
        placed = station.toggle_pitcher()
        task.notify_world_changed()
        safety.note("Milk pitcher placed in the carry path." if placed else "Milk pitcher removed.")

    env.add(state_label)
    env.add(task_label)
    env.add(swift.Button(cb=lambda _=None: safety.press_estop("GUI"), desc="E-STOP"))
    env.add(swift.Button(cb=lambda _=None: safety.release_estop(), desc="Release E-stop"))
    env.add(swift.Button(cb=lambda _=None: safety.reset(), desc="RESET"))
    env.add(swift.Button(cb=lambda _=None: safety.start(), desc="START / RESUME"))
    env.add(swift.Button(cb=toggle_hand, desc="Hand into light curtain (toggle)"))
    env.add(swift.Button(cb=toggle_pitcher, desc="Milk pitcher in path (toggle)"))

    task.update_visuals()
    env.step(0)
    if args.autostart:
        safety.start()

    print("Swift running. Use the panel buttons; Ctrl+C here to exit.")
    last_status = None
    try:
        # One iteration per frame. env.step(DT) paces the loop in real time
        # (it sleeps) so nothing here spins the processor while stopped.
        while True:
            safety.tick()
            hand = [cell.hand_obstacle] if cell.hand_inside else []
            safety.update_curtain(curtain.is_broken(hand))
            task.tick()
            task.update_visuals()
            if task.status != last_status:
                last_status = task.status
                task_label.desc = f"Task: {task.status}"
            env.step(DT)
    except KeyboardInterrupt:
        print("Exiting.")


if __name__ == "__main__":
    main()
