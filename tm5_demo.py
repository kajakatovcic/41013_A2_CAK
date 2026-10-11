"""
OMRON TM5-700 plate station in the shared workcell

    python tm5_demo.py                          # Swift demo, press START in the panel
    python tm5_demo.py --plan                   # no Swift: plan every move + print
                                                # the jtraj vs RMRC comparison

Sequence: An empty tray arrives from the upper end of the conveyor belt to the TM5 
station. During this, the TM5 lowers its 'suction cup' ont othe top plate in the 
dispenser, lifts it straight up between the guide posts, carries it level to the 
tray and lowers it into the tray's plate slot.
The conveyor then takes the tray on to the collection zone. 
NOTE: currnetly in main_scene.py it stops at the JAKA as the other two bots have
not yet been implemented.

The FINAL JOINT STATE can be set by FINAL_Q_DEG in tm5_task.py
"""

import argparse

import numpy as np
from ir_support_extra_robots.robots import OmronTM5700

from tm5_motion import TM5Planner
from tm5_station import PlateStation
from tm5_task import TM5Task
from safety_controller import SafetyController
from workcell_scene import Workcell
from workcell_layout import STATIONS, Z_COUNTER, TRAY_ENTRY_X
import cell_runner
import plan_report

# Link frames must stay 5 cm above the counter (the TM5's wrist is larger
# than the JAKA's).
MIN_LINK_Z = Z_COUNTER + 0.05

# The moves compared in the --plan report: (label, start, goal, RMRC mode, holding)
COMPARE_CASES = [
    ("lower cup onto plate", "above_stack", "pick", "line", False),
    ("lift plate from stack", "pick", "above_stack", "line", True),
    ("carry plate to tray", "above_stack", "above_tray", "arc", True),
    ("lower plate onto tray", "above_tray", "on_tray", "line", True),
]


def build_tm5(obstacles=None):
    """The TM5-700, its plate station and its planner (also used by main_scene.py)."""
    base = STATIONS["TM5700"][0]
    robot = OmronTM5700(base=base)
    station = PlateStation(base)
    planner = TM5Planner(robot, station.obstacles if obstacles is None else obstacles, MIN_LINK_Z)
    return robot, station, planner


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
    cell_runner.add_controls(env, safety, cell, state_label, task_label)
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), task, task_label)


if __name__ == "__main__":
    main()
