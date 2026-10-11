"""
Smooth Oper-Caterer: Full workcell with all robots integrated (eventually)

    python main_scene.py                        # Swift, press START in the panel
    python main_scene.py --plan                 # no Swift: plan every robot's moves

FINAL JOINT STATES: each robot's final pose is set by FINAL_Q_DEG at the top
of its job file (degrees). Change them there:
    JAKA MiniCobo -> coffee_task.py
    TM5-700       -> tm5_task.py

Order flow along the conveyor (one tray):
    1. TM5-700  places a plate (tray stops at the TM5)
    2. Lynxmotion  serves food onto the plate @TODO when integrated
    3. FAIRINO FR3 places a drink @TODO when integrated
    4. JAKA MiniCobo brews and places the coffee (tray stops at the JAKA)
    5. The tray moves out through the light curtain to the collection zone

Every robot runs at once in its own lane of the shared TaskSequencer. They
only wait for each other through the tray. The JAKA picks a cup and brews
while the tray isnt yet at its station and only carries the coffee to the tray
once the tray has arrived. One safety controller stops every robot and the
conveyor together.

NOTE for TEAMMATES :)
TO ADD A ROBOT: build it with its station and planner (see build_tm5 in
tm5_demo.py), give it a job with segments that wait for 'tray_at_<name>'
and set '<name>_placed', add its RobotLane below, and add its stop to the
conveyor lane in order.
"""

import argparse

import numpy as np

from coffee_demo import build_jaka, pitcher_toggle
from coffee_task import CoffeeJob
from tm5_demo import build_tm5
from tm5_task import PlateJob
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
    jaka, jaka_station, jaka_planner = build_jaka()
    tm5.q = np.zeros(6)
    jaka.q = np.zeros(6)

    plate_job = PlateJob(tm5_station, tray)
    coffee_job = CoffeeJob(jaka_station, tray)

    S = Segment
    conveyor = [
        S("convey", "tray -> TM5 station", x=plate_job.tray_stop_x, sets="tray_at_tm5"),
        # @TODO: Lynxmotion and FR3 stops go here in order along the belt.
        S("convey", "tray -> JAKA station", x=coffee_job.tray_stop_x,
          requires="plate_placed", sets="tray_at_jaka"),
        S("convey", "tray -> collection zone", x=COLLECTION_X,
          requires="coffee_placed", sets="delivered"),
    ]
    # Each robot's sequence ends with its final joint state, set by
    # FINAL_Q_DEG in coffee_task.py (JAKA) and tm5_task.py (TM5).
    lanes = [
        RobotLane("TM5", tm5, tm5_planner, plate_job, plate_job.segments()),
        RobotLane("JAKA", jaka, jaka_planner, coffee_job, coffee_job.segments()),
        ConveyorLane("conveyor", tray, conveyor),
    ]
    sequencer = TaskSequencer(safety, lanes)
    parts = [tm5_station, tm5, jaka_station, jaka]
    return cell, sequencer, parts, jaka_station


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
    cell, sequencer, parts, jaka_station = build_scene(None)
    env = cell_runner.launch(cell, parts, ([0.0, 4.2, 3.6], [-0.9, -0.6, 0.8]))
    safety, state_label, task_label = cell_runner.make_safety(cell)
    sequencer.safety = safety
    cell_runner.add_controls(env, safety, cell, state_label, task_label,
                             [("Milk pitcher in JAKA path (toggle)",
                               pitcher_toggle(jaka_station, sequencer, safety))])
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), sequencer, task_label)


if __name__ == "__main__":
    main()
