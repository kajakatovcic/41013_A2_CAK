"""
FAIRINO FR3 cold-drink station in the shared workcell.

    python fairino_fr3_demo.py
        Swift demo - press START in the side panel

    python fairino_fr3_demo.py --plan
        No Swift - plan every move and print the sequence
"""

import argparse
import numpy as np
from spatialmath import SE3

from fairino_fr3 import FairinoFR3
from fairino_fr3_motion import FR3Planner
from fairino_fr3_station import FR3Station
from fairino_fr3_task import FR3Task

from safety_controller import SafetyController
from workcell_scene import Workcell
from workcell_layout import (
    STATIONS,
    Z_COUNTER,
    TRAY_ENTRY_X
)

import cell_runner
import plan_report


# -----------------------------------------------------------------------------#
# Planner clearance
# -----------------------------------------------------------------------------#

# Start with the same general clearance used by the smaller cobots.
MIN_LINK_Z = Z_COUNTER + 0.03


# -----------------------------------------------------------------------------#
# Build FAIRINO FR3 station
# -----------------------------------------------------------------------------#

def build_fr3(obstacles=None):

    # Temporary validated position for the standalone FR3 demo.
    station_base = STATIONS["FAIRINO_FR3"][0]

    robot_base = station_base * SE3(0, 0.15, 0)
    robot = FairinoFR3(base=robot_base)

    station = FR3Station(station_base, robot_base)

    planner = FR3Planner(
        robot,
        station.obstacles if obstacles is None else obstacles,
        MIN_LINK_Z
    )

    return robot, station, planner


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--plan",
        action="store_true",
        help="plan and report only, no Swift"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Build robot / station / planner
    # ---------------------------------------------------------

    robot, station, planner = build_fr3()

    # Use the FR3's known working home pose
    robot.q = np.array(
        robot.home_q,
        dtype=float
    )

    # ---------------------------------------------------------
    # Shared workcell
    # ---------------------------------------------------------

    # Empty tray:
    # FR3 will place the real drink onto it.
    cell = Workcell(
        tray_x=TRAY_ENTRY_X,
        tray_placeholders=()
    )

    # ---------------------------------------------------------
    # PLAN ONLY
    # ---------------------------------------------------------

    if args.plan:

        safety = SafetyController()

        task = FR3Task(
            robot,
            planner,
            station,
            cell.conveyor.tray,
            safety
        )

        plan_report.print_plan(
            task,
            task.lanes[0],
            "PLANNED FAIRINO FR3 SEQUENCE "
            "(each move planned from the end of the previous)"
        )

        return

    # ---------------------------------------------------------
    # SWIFT DEMO
    # ---------------------------------------------------------

    env = cell_runner.launch(
        cell,
        [station, robot],
        (
            [1.6, 1.8, 2.0],
            [0.0, -0.5, 0.8]
        )
    )

    safety, state_label, task_label = (
        cell_runner.make_safety(cell)
    )

    task = FR3Task(
        robot,
        planner,
        station,
        cell.conveyor.tray,
        safety
    )

    cell_runner.add_controls(
        env,
        safety,
        cell,
        state_label,
        task_label
    )

    cell_runner.run(
        env,
        safety,
        cell,
        cell_runner.make_curtain(),
        task,
        task_label
    )


if __name__ == "__main__":
    main()