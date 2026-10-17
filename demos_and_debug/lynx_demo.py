"""
Lynxmotion SES-PRO cookie station in the shared workcell.

    python demos_and_debug/lynx_demo.py          # Swift demo, press START in the panel
    python demos_and_debug/lynx_demo.py --plan   # no Swift: plan every move + print
                                                 # the jtraj vs RMRC comparison

Sequence: an empty tray rides in from the upstream end of the conveyor to the
Lynxmotion station. Meanwhile the Lynxmotion lowers its gripper onto a cookie
on the board, lifts it, carries it level round to the tray and places it.
The conveyor then takes the tray on to the collection zone. (In main_scene.py
the plate is placed first by the TM5-700 and the cookie goes on the plate.)

Swift panel controls: the shared safety panel (cell_runner.py), plus
    Cookie jar in path - deliberately place an object in the cookie's carry path

The FINAL JOINT STATE is set by FINAL_Q_DEG in lynx_task.py.
"""

import os
import sys
# This script lives in demos_and_debug/: put the repo root (one level up) on
# the import path so the shared workcell modules are found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from lynx_station import build_lynx
from lynx_task import LynxTask, COMPARE_CASES
from safety_controller import SafetyController
from workcell_scene import Workcell
from workcell_layout import TRAY_ENTRY_X
import cell_runner
import plan_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="plan and report only, no Swift")
    args = parser.parse_args()

    robot, station, planner = build_lynx()

    cell = Workcell(tray_x=TRAY_ENTRY_X, tray_placeholders=())
    robot.q = np.zeros(6)

    if args.plan:
        task = LynxTask(robot, planner, station, cell.conveyor.tray, SafetyController())
        plan_report.compare_methods(planner, task.job, COMPARE_CASES)
        plan_report.print_plan(task, task.lanes[0], "PLANNED LYNXMOTION SEQUENCE")
        return

    env = cell_runner.launch(cell, [station], ([-0.5, 2.6, 2.3], [-1.7, -0.5, 0.9]))
    safety, state_label, task_label = cell_runner.make_safety(cell)
    task = LynxTask(robot, planner, station, cell.conveyor.tray, safety)
    cell_runner.add_controls(env, safety, cell, state_label, task_label,
                             [cell_runner.obstacle_button("Cookie jar in path (toggle)",
                                                          station.toggle_jar, task, safety,
                                                          "Cookie jar")])
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), task, task_label)


if __name__ == "__main__":
    main()