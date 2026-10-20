# UR3e Plate Task
#
#   robot: ready -> suction-pick the top plate -> carry level -> place on tray -> ready
#   conveyor: bring the empty tray to the UR3e -> (wait for the plate) -> keep moving
#
# PlateJob is the UR3e station's part of the task (its segments and suction
# actions). It connects to the shared TaskSequencer.
# The lanes synchronise via flags. i.e. the carry to the tray waits for
# 'tray_at_ur3e' and the conveyor waits for 'plate_placed'.

import numpy as np
from spatialmath import SE3

from ur3e_motion import TOOL
from ur3e_station import POSES, IK_SEEDS, READY_Q
from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from workcell_layout import COLLECTION_X
from workcell_scene import Tray

SUCTION_FRAMES = 8 # frames for realistic suction


# Can supply final joint state as a q in degrees 
# used in both demos_and_debug/ur3e_demo.py and main_scene.py
# If set to None the sequences ends with the UR3e at a ready pose
# A goal outside the limits stops the cell with a fault. A goal whose direct
# path would collide is re-planned through a raised via-point.
FINAL_Q_DEG = [0, -90, 0, -90, 0, 0] # UR 'L' pose: upper arm up, forearm level, clear of the belt


class PlateJob:
    """The UR3e plate station's job for the shared task sequencer."""

    def __init__(self, station, tray):
        self.station = station
        self.tray = tray
        self.plate = station.top_plate
        # Conveyor stop for the UR3e. The tray halts with the "PLATE_SLOT"
        # exactly at the planner's 'on_tray' target in the UR3e base frame
        r, phi, _ = POSES["on_tray"]
        slot_x = (station.base * SE3(r * np.cos(phi), r * np.sin(phi), 0)).t[0]
        self.tray_stop_x = slot_x - Tray.PLATE_SLOT.t[0]

    def segments(self):
        S = Segment
        segs = [
            S("jtraj", "home -> ready", goal=READY_Q),
            S("jtraj", "ready -> above plate stack", goal="above_stack"),
            S("rmrc", "lower suction cup onto plate", goal="pick", mode="line"),
            S("suction", "suction on", on=True, hold=True),
            S("rmrc", "lift plate between guide posts", goal="above_stack", mode="line"),
            S("rmrc", "carry plate to tray (level arc)", goal="above_tray", mode="arc",
              requires="tray_at_ur3e"),
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

    def report_start(self, name):
        """Where the --plan comparison starts its IK from. From all-zero joints
        ikine_LM lands the UR3e on another branch (wrist flipped), which
        the real sequence never uses since it always starts from READY_Q."""
        return READY_Q

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


class UR3eTask(TaskSequencer):
    """The UR3e station on its own. UR3e lane + conveyor lane (demos_and_debug/ur3e_demo.py)"""

    def __init__(self, robot, planner, station, tray, safety, log=print):
        self.tray = tray
        self.job = PlateJob(station, tray)
        self.tray_stop_x = self.job.tray_stop_x
        S = Segment
        conveyor = [
            S("convey", "tray -> UR3e station", x=self.tray_stop_x, sets="tray_at_ur3e"),
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="plate_placed",
              sets="delivered"),
        ]
        lanes = [RobotLane("UR3e", robot, planner, self.job, self.job.segments()),
                 ConveyorLane("conveyor", tray, conveyor)]
        super().__init__(safety, lanes, log)

    def dry_run(self, lane=None):
        return super().dry_run(lane or self.lanes[0])
