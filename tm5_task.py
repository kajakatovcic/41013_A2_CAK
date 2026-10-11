# TM5-700 Plate Task
#
#   robot: ready -> suction-pick the top plate -> carry level -> place on tray -> ready
#   conveyor: bring the empty tray to the TM5 -> (wait for the plate) -> keep moving
#
# PlateJob is the TM5 station's part of the task (its segments and suction
# actions). It connects to the shared TaskSequencer, so the same job runs in
# tm5_demo.py on its own and in main_scene.py with every robot.
# The lanes synchronise via flags. i.e. the carry to the tray waits for
# 'tray_at_tm5' and the conveyor waits for 'plate_placed'.

import numpy as np
from spatialmath import SE3

from tm5_motion import TOOL
from tm5_station import POSES, IK_SEEDS
from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from workcell_layout import COLLECTION_X
from workcell_scene import Tray

SUCTION_FRAMES = 8 # frames for realistic suction


# Can supply final joint state as a q in degrees 
# used in both tm5_demo and main_scene
# If set to None the sequences ends with the TM5 at a ready pose
# Joint limits: J1 +-270, J2 +-180, J3 +-155, J4 +-180, J5 +-180, J6 +-270
# A goal outside the limits stops the cell with a fault. A goal whose direct
# path would collide is re-planned through a raised via-point.
FINAL_Q_DEG = [135, 60, -90, 30, 90, 0]


class PlateJob:
    """The TM5 plate station's job for the shared task sequencer."""

    def __init__(self, station, tray):
        self.station = station
        self.tray = tray
        self.plate = station.top_plate
        # Conveyor stop for the TM5. The tray halts with the "PLATE_SLOT"
        # exactly at the planner's 'on_tray' target in the TM5 base frame
        r, phi, _ = POSES["on_tray"]
        slot_x = (station.base * SE3(r * np.cos(phi), r * np.sin(phi), 0)).t[0]
        self.tray_stop_x = slot_x - Tray.PLATE_SLOT.t[0]

    def segments(self):
        S = Segment
        segs = [
            S("jtraj", "home -> ready", goal="ready"),
            S("jtraj", "ready -> above plate stack", goal="above_stack"),
            S("rmrc", "lower suction cup onto plate", goal="pick", mode="line"),
            S("suction", "suction on", on=True, hold=True),
            S("rmrc", "lift plate between guide posts", goal="above_stack", mode="line"),
            S("rmrc", "carry plate to tray (level arc)", goal="above_tray", mode="arc",
              requires="tray_at_tm5"),
            S("rmrc", "lower plate onto tray", goal="on_tray", mode="line"),
            S("suction", "suction off", on=False, hold=False),
            S("rmrc", "lift suction cup clear", goal="clear_tray", mode="line",
              sets="plate_placed"),
            S("jtraj", "return to ready", goal="ready"),
        ]
        if FINAL_Q_DEG is not None: # set at tje top
            segs.append(S("jtraj", "final joint state (supplied)", goal=np.deg2rad(FINAL_Q_DEG)))
        return segs

    # Targets
    def target_pose(self, name):
        return self.station.tcp_pose(name)

    def target_polar(self, name):
        return POSES[name]

    def ik_seeds(self, name):
        """IK_SEEDS with joint 1 turned to face the target's bearing"""
        seeds = []
        for s in IK_SEEDS:
            s = s.copy()
            s[0] = POSES[name][1]
            seeds.append(s)
        return seeds

    # Non-motion segments
    def run_action(self, seg, lane):
        if seg.kind == "suction":
            seg.i += 1
            finished = seg.i >= SUCTION_FRAMES
            if finished and not seg.p["on"]:
                # Put the plate on the tray at its exact current pose
                self.tray.attach(self.plate, self.tray.pose.inv() * self.plate.pose)
            return finished
        return True

    def update_visuals(self, lane):
        """Suction tool and held plate follow the flange for every frame in the scene"""
        T_flange = lane.robot.fkine(lane.robot.q)
        self.station.tool.update(T_flange)
        if lane.holding:
            self.plate.held_by(T_flange * TOOL)


class TM5Task(TaskSequencer):
    """The TM5 station on its own - TM5 lane + conveyor lane (tm5_demo.py)"""

    def __init__(self, robot, planner, station, tray, safety, log=print):
        self.tray = tray
        self.job = PlateJob(station, tray)
        self.tray_stop_x = self.job.tray_stop_x
        S = Segment
        conveyor = [
            S("convey", "tray -> TM5 station", x=self.tray_stop_x, sets="tray_at_tm5"),
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="plate_placed",
              sets="delivered"),
        ]
        lanes = [RobotLane("TM5", robot, planner, self.job, self.job.segments()),
                 ConveyorLane("conveyor", tray, conveyor)]
        super().__init__(safety, lanes, log)

    def dry_run(self, lane=None):
        return super().dry_run(lane or self.lanes[0])
