"""
OMRON TM5-700 plate station in the shared workcell

    python demos_and_debug/tm5_demo.py          # Swift demo, press START in the panel
    python demos_and_debug/tm5_demo.py --plan   # no Swift: plan every move + print
                                                # the jtraj vs RMRC comparison

Sequence: An empty tray arrives from the upper end of the conveyor belt to the TM5 
station. During this, the TM5 lowers its 'suction cup' ont othe top plate in the 
dispenser, lifts it straight up between the guide posts, carries it level to the 
tray and lowers it into the tray's plate slot.
The conveyor then takes the tray on to the collection zone. In main_scene.py
the tray instead continues to the Lynxmotion (food) and then the JAKA (coffee).

The FINAL JOINT STATE can be set by FINAL_Q_DEG in tm5_task.py
"""

import os
import sys
# This script lives in demos_and_debug/: put the repo root (one level up) on
# the import path so the shared workcell modules are found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from tm5_station import build_tm5
from tm5_task import TM5Task
from safety_controller import SafetyController
from workcell_scene import Workcell
from workcell_layout import TRAY_ENTRY_X
import cell_runner
import plan_report
from teach_pendant import TeachPendant

# The moves compared in the --plan report: (label, start, goal, RMRC mode, holding)
COMPARE_CASES = [
    ("lower cup onto plate", "above_stack", "pick", "line", False),
    ("lift plate from stack", "pick", "above_stack", "line", True),
    ("carry plate to tray", "above_stack", "above_tray", "arc", True),
    ("lower plate onto tray", "above_tray", "on_tray", "line", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="plan and report only, no Swift")
    args = parser.parse_args()

    robot, station, planner = build_tm5()
    robot.q = np.zeros(6)
    # The tray starts empty: the TM5 places the real plate.
    cell = Workcell(tray_x=TRAY_ENTRY_X, tray_placeholders=())

    if args.plan:
        task = TM5Task(robot, planner, station, cell.conveyor.tray, SafetyController())
        plan_report.compare_methods(planner, task.job, COMPARE_CASES)
        plan_report.print_plan(task, task.lanes[0],
                               "PLANNED TM5 SEQUENCE (each move planned from the end of the previous)")
        return

    env = cell_runner.launch(cell, [station, robot], ([-1.9, 1.6, 2.2], [-3.4, -0.4, 0.9]))
    safety, state_label, task_label = cell_runner.make_safety(cell)
    # The TM5's final joint state comes from FINAL_Q_DEG in tm5_task.py.
    task = TM5Task(robot, planner, station, cell.conveyor.tray, safety)
    pendant = TeachPendant(task, safety) # GUI jog, joystick and physical e-stop
    cell_runner.add_controls(env, safety, cell, state_label, task_label, pendant=pendant)
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), task, task_label,
                    pendant=pendant)


if __name__ == "__main__":
    main()
