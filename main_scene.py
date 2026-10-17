"""
Smooth Oper-Caterer: Full workcell with every integrated robot

    python main_scene.py                        # Swift, press START in the panel
    python main_scene.py --plan                 # no Swift: plan every robot's moves

FINAL JOINT STATES: each robot's final pose is set by FINAL_Q_DEG at the top
of its job file (degrees). Change them there:
    TM5-700            -> tm5_task.py
    Lynxmotion SES-PRO -> lynx_task.py
    JAKA MiniCobo      -> coffee_task.py

Order flow along the conveyor (one tray):
    1. TM5-700  places a plate (tray stops at the TM5)
    2. Lynxmotion SES-PRO places a cookie on the plate (tray stops at the Lynxmotion)
    3. FAIRINO FR3 places a drink @TODO when integrated
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

Each station also runs on its own from demos_and_debug/(coffee_demo.py,
tm5_demo.py, lynx_demo.py).

NOTE for TEAMMATES :)
TO ADD A ROBOT: build it with its station and planner (see build_tm5 in
tm5_station.py), give it a job with segments that wait for 'tray_at_<name>'
and set '<name>_placed', add its RobotLane below, and add its stop to the
conveyor lane in order.
"""

import argparse

import numpy as np

from coffee_station import build_jaka
from coffee_task import CoffeeJob
from tm5_station import build_tm5
from tm5_task import PlateJob
from lynx_station import build_lynx
from lynx_task import LynxJob
from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from safety_controller import SafetyController
from workcell_scene import Workcell
from workcell_layout import TRAY_ENTRY_X, COLLECTION_X
import cell_runner
import plan_report


def build_scene(safety):
    """Every integrated robot, its station, and one sequencer which drives all robots."""
    # The tray starts empty at the upstream end. The TM5 places the plate.
    cell = Workcell(tray_x=TRAY_ENTRY_X, tray_placeholders=())
    tray = cell.conveyor.tray

    tm5, tm5_station, tm5_planner = build_tm5()
    lynx, lynx_station, lynx_planner = build_lynx()
    jaka, jaka_station, jaka_planner = build_jaka()
    tm5.q = np.zeros(6)
    lynx.q = np.zeros(6)
    jaka.q = np.zeros(6)

    plate_job = PlateJob(tm5_station, tray)
    lynx_job = LynxJob(lynx_station, tray)
    coffee_job = CoffeeJob(jaka_station, tray)

    S = Segment
    conveyor = [
        S("convey", "tray -> TM5 station", x=plate_job.tray_stop_x, sets="tray_at_tm5"),
        S("convey", "tray -> Lynxmotion station", x=lynx_job.tray_stop_x,
          requires="plate_placed", sets="tray_at_lynx"),
        # @TODO: the FR3 stop goes here (requires="food_placed",
        # sets="tray_at_fr3"). Then change the JAKA stop to requires="drink_placed".
        S("convey", "tray -> JAKA station", x=coffee_job.tray_stop_x,
          requires="food_placed", sets="tray_at_jaka"),
        S("convey", "tray -> collection zone", x=COLLECTION_X,
          requires="coffee_placed", sets="delivered"),
    ]
    # Each robot's sequence ends with its final joint state, set by
    # FINAL_Q_DEG in tm5_task.py (TM5), lynx_task.py (Lynxmotion) and
    # coffee_task.py (JAKA).
    lanes = [
        RobotLane("TM5", tm5, tm5_planner, plate_job, plate_job.segments()),
        RobotLane("Lynxmotion", lynx, lynx_planner, lynx_job, lynx_job.segments()),
        RobotLane("JAKA", jaka, jaka_planner, coffee_job, coffee_job.segments()),
        ConveyorLane("conveyor", tray, conveyor),
    ]
    sequencer = TaskSequencer(safety, lanes)
    # The Lynxmotion has no add_to_env of its own. Its station draws and moves
    # its link meshes, so only the station is added to Swift.
    parts = [tm5_station, tm5, lynx_station, jaka_station, jaka]
    stations = {"TM5": tm5_station, "Lynxmotion": lynx_station, "JAKA": jaka_station}
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
    cell_runner.add_controls(env, safety, cell, state_label, task_label, [
        cell_runner.obstacle_button("Milk pitcher in JAKA path (toggle)",
                                    stations["JAKA"].toggle_pitcher, sequencer, safety,
                                    "Milk pitcher"),
        cell_runner.obstacle_button("Cookie jar in Lynxmotion path (toggle)",
                                    stations["Lynxmotion"].toggle_jar, sequencer, safety,
                                    "Cookie jar"),
    ])
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), sequencer, task_label)


if __name__ == "__main__":
    main()
