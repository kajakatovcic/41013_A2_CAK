"""
Smooth Oper-Caterer: Full workcell with every integrated robot

    python main_scene.py                        # Swift, press START in the panel
    python main_scene.py --plan                 # no Swift: plan every robot's moves

FINAL JOINT STATES: each robot's final pose is set by FINAL_Q_DEG at the top
of its job file (degrees). Change them there:
    UR3e               -> ur3e_task.py
    Lynxmotion SES-PRO -> lynx_task.py
    Fairino FR3        -> fairino_fr3_task.py
    JAKA MiniCobo      -> coffee_task.py

Order flow along the conveyor (one tray):
    1. UR3e places a plate (tray stops at the UR3e)
    2. Lynxmotion SES-PRO places a cookie on the plate (tray stops at the Lynxmotion)
    3. FAIRINO FR3 places a cold drink - soda can (tray stops at the Fairino FR3)
    4. JAKA MiniCobo brews and places the coffee (tray stops at the JAKA)
    5. The tray moves out through the light curtain to the collection zone

Every robot runs at once in its own lane of the shared TaskSequencer. They
only wait for each other through the tray. The JAKA picks a cup and brews
while the tray isnt yet at its station and only carries the coffee to the tray
once the tray has arrived. One safety controller stops every robot and the
conveyor together.

Swift panel: the shared safety panel (cell_runner.py), plus a movable
obstacle for each robot that has one (JAKA milk pitcher, Lynxmotion cookie jar)
to deliberately block a planned path.

TEACH/JOG (teach_pendant.py) below the safety controls:
    'AUTO/MANUAL' button toggles the mode. In MANUAL the task sequence and
    conveyor pause and the selected robot is jogged with the joint sliders, or
    in X/Y/Z with a joystick (left stick = X/Y, Triangle = Z up, Circle = Z
    down). The joystick rumbles at joint limits.
    Joystick CROSS (X) button = PHYSICAL E-STOP, active in every mode.

Each station also runs on its own from demos_and_debug/(coffee_demo.py,
ur3e_demo.py, lynx_demo.py, fairino_fr3_demo.py).

NOTE for TEAMMATES :)
TO ADD A ROBOT: build it with its station and planner (see build_ur3e in
ur3e_station.py), give it a job with segments that wait for 'tray_at_<name>'
and set '<name>_placed', add its RobotLane below, and add its stop to the
conveyor lane in order.
"""

import argparse

import numpy as np

from coffee_station import build_jaka
from coffee_task import CoffeeJob
from ur3e_station import build_ur3e
from ur3e_task import PlateJob
from lynx_station import build_lynx
from lynx_task import LynxJob
from fairino_fr3_station import build_fr3
from fairino_fr3_task import FR3Job
from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from safety_controller import SafetyController
from workcell_scene import Workcell
from workcell_layout import TRAY_ENTRY_X, COLLECTION_X
import cell_runner
import plan_report
from teach_pendant import TeachPendant


def build_scene(safety):
    """Every integrated robot, its station, and one sequencer which drives all robots."""
    # The tray starts empty at the upstream end. The UR3e places the plate.
    cell = Workcell(tray_x=TRAY_ENTRY_X, tray_placeholders=())
    tray = cell.conveyor.tray

    ur3e, ur3e_station, ur3e_planner = build_ur3e()
    lynx, lynx_station, lynx_planner = build_lynx()
    jaka, jaka_station, jaka_planner = build_jaka()
    fr3, fr3_station, fr3_planner = build_fr3()
    ur3e.q = np.array(ur3e.home_q, dtype=float) # arm up, as for the FR3
    lynx.q = np.zeros(6)
    fr3.q = np.array(fr3.home_q, dtype=float)
    jaka.q = np.zeros(6)

    plate_job = PlateJob(ur3e_station, tray)
    lynx_job = LynxJob(lynx_station, tray)
    fr3_job = FR3Job(fr3_station, tray)
    coffee_job = CoffeeJob(jaka_station, tray)

    S = Segment
    conveyor = [
        S("convey", "tray -> UR3e station", x=plate_job.tray_stop_x, sets="tray_at_ur3e"),
        S("convey", "tray -> Lynxmotion station", x=lynx_job.tray_stop_x,
          requires="plate_placed", sets="tray_at_lynx"),
        S("convey", "tray -> FAIRINO FR3 station", x=fr3_job.tray_stop_x,
          requires="food_placed", sets="tray_at_fr3"),
        S("convey", "tray -> JAKA station", x=coffee_job.tray_stop_x,
          requires="drink_placed", sets="tray_at_jaka"),
        S("convey", "tray -> collection zone", x=COLLECTION_X,
          requires="coffee_placed", sets="delivered"),
    ]
    # Each robot's sequence ends with its final joint state set by
    # FINAL_Q_DEG in ur3e_task.py (UR3e), lynx_task.py (Lynxmotion),
    # fairino_fr3_task.py (FR3) and coffee_task.py (JAKA).
    lanes = [
        RobotLane("UR3e", ur3e, ur3e_planner, plate_job, plate_job.segments()),
        RobotLane("Lynxmotion", lynx, lynx_planner, lynx_job, lynx_job.segments()),
        RobotLane("FAIRINO FR3", fr3, fr3_planner, fr3_job, fr3_job.segments()),
        RobotLane("JAKA", jaka, jaka_planner, coffee_job, coffee_job.segments()),
        ConveyorLane("conveyor", tray, conveyor),
    ]
    sequencer = TaskSequencer(safety, lanes)
    # The Lynxmotion has no add_to_env of its own. Its station draws and moves
    # its link meshes, so only the station is added to Swift.
    parts = [ur3e_station, ur3e, lynx_station, fr3_station, fr3, jaka_station, jaka]
    stations = {"UR3e": ur3e_station, "Lynxmotion": lynx_station, "FAIRINO_FR3": fr3_station, "JAKA": jaka_station}
    return cell, sequencer, parts, stations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="plan and report only, no Swift")
    args = parser.parse_args()

    if args.plan:
        cell, sequencer, parts, _ = build_scene(SafetyController())
        for lane in sequencer.robot_lanes:
            plan_report.print_plan(sequencer, lane, f"PLANNED {lane.name} SEQUENCE")
        return

    # The safety controller needs the cell's indicators and the sequencer
    # needs the safety controller, so the scene is built around a placeholder
    # first and the real controller is attached before anything runs.
    cell, sequencer, parts, stations = build_scene(None)
    env = cell_runner.launch(cell, parts, ([0.0, 4.2, 3.6], [-0.9, -0.6, 0.8]))
    safety, state_label, task_label = cell_runner.make_safety(cell)
    sequencer.safety = safety
    pendant = TeachPendant(sequencer, safety) # GUI jog, joystick and physical e-stop
    cell_runner.add_controls(env, safety, cell, state_label, task_label, [
        cell_runner.obstacle_button("Milk pitcher in JAKA path (toggle)",
                                    stations["JAKA"].toggle_pitcher, sequencer, safety,
                                    "Milk pitcher"),
        cell_runner.obstacle_button("Cookie jar in Lynxmotion path (toggle)",
                                    stations["Lynxmotion"].toggle_jar, sequencer, safety,
                                    "Cookie jar"),
        cell_runner.obstacle_button("Blue crate in FR3 path (toggle)",
                                stations["FAIRINO_FR3"].toggle_obstacle_crate,
                                sequencer, safety,
                                "Blue crate"),       
    ], pendant=pendant)
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), sequencer, task_label,
                    pendant=pendant)


if __name__ == "__main__":
    main()
