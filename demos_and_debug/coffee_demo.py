"""
Smooth Oper-Caterer: JAKA MiniCobo coffee station in the shared workcell.

    python demos_and_debug/coffee_demo.py           # Swift demo, press START in the panel
    python demos_and_debug/coffee_demo.py --plan    # no Swift: plan every move + print
                                                    # the jtraj vs RMRC comparison

The final pose is set by FINAL_Q_DEG at the top of coffee_task.py (degrees). 
Change it there. Used by both demo and main scene

Swift panel controls (the simulated GUI e-stop and safety inputs):
    E-STOP / Release E-stop / RESET / START-RESUME
    Hand into light curtain   - asynchronous unsafe-zone signal at the exit
    Milk pitcher in path      - deliberately place an object in the carry path

Sequence: the tray (already holding the plate, food and drink from the
upstream stations) rides to the JAKA. Meanwhile the JAKA takes a cup from the
dispenser, brews a coffee, carries it level to the tray, and the conveyor takes
the finished order out through the light curtain to the collection zone.
"""

import os
import sys
# This script lives in demos_and_debug/: put the repo root (one level up) on
# the import path so the shared workcell modules are found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from coffee_station import build_jaka, POSES, IK_SEEDS
from coffee_task import CoffeeTask
from safety_controller import SafetyController
from workcell_scene import Workcell
import cell_runner
import plan_report
from teach_pendant import TeachPendant


def build():
    robot, station, planner = build_jaka()
    cell = Workcell()
    curtain = cell_runner.make_curtain()
    return robot, station, cell, planner, curtain


def ik_at(planner, station, name, q_hint):
    seeds = []
    for s in IK_SEEDS:
        s = s.copy()
        s[0] = POSES[name][1]
        seeds.append(s)
    return planner.solve_ik(station.tcp_pose(name), q_hint, seeds)


# The moves compared in the --plan report: (label, start, goal, RMRC mode, holding)
COMPARE_CASES = [
    ("approach cup on pad", "pre_pick", "pick", "line", False),
    ("insert cup into bay", "pre_insert", "insert", "line", True),
    ("carry full cup to tray", "pre_insert", "above_tray", "arc", True),
    ("slide fingers off cup", "on_tray", "retreat", "line", True),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true", help="plan and report only, no Swift")
    args = parser.parse_args()

    robot, station, cell, planner, curtain = build()
    robot.q = np.zeros(6)

    if args.plan:
        safety = SafetyController()
        task = CoffeeTask(robot, planner, station, cell.conveyor.tray, safety)
        plan_report.compare_methods(planner, task.job, COMPARE_CASES)
        plan_report.print_plan(task, task.lanes[0],
                               "PLANNED JAKA SEQUENCE (each move planned from the end of the previous)")
        return

    env = cell_runner.launch(cell, [station, robot], ([3.4, 2.6, 2.3], [1.4, -0.5, 0.9]))
    safety, state_label, task_label = cell_runner.make_safety(cell)
    # The JAKA's final joint state comes from FINAL_Q_DEG in coffee_task.py.
    task = CoffeeTask(robot, planner, station, cell.conveyor.tray, safety)
    pendant = TeachPendant(task, safety) # GUI jog, joystick and physical e-stop
    cell_runner.add_controls(env, safety, cell, state_label, task_label,
                             [cell_runner.obstacle_button("Milk pitcher in path (toggle)",
                                                          station.toggle_pitcher, task, safety,
                                                          "Milk pitcher")],
                             pendant=pendant)
    cell_runner.run(env, safety, cell, curtain, task, task_label, pendant=pendant)


if __name__ == "__main__":
    main()
