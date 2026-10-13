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
    cell_runner.add_controls(env, safety, cell, state_label, task_label, [])
    cell_runner.run(env, safety, cell, cell_runner.make_curtain(), task, task_label)


if __name__ == "__main__":
    main()