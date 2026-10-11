# The JAKA coffee order.
#
#   robot: ready -> pick cup -> brew -> carry -> place on tray -> ready
#   conveyor: bring the tray to the JAKA -> (wait for coffee) -> collection
#
# CoffeeJob is the JAKA station's part of the task (its segments and its
# gripper/brew actions). It works into into the shared TaskSequencer so the same
# job runs in coffee_demo.py on its own and in main_scene.py with every robot.
# The lanes are synchronised via FLAGS. i.e. the carry to the tray waits for
# 'tray_at_jaka' and the conveyor waits for 'coffee_placed'.

import numpy as np
from spatialmath import SE3

from jaka_motion import DT, TOOL
from coffee_station import POSES, IK_SEEDS, Gripper
from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from workcell_layout import COLLECTION_X
from workcell_scene import Tray


# Final JAKA joint state
# a jtraj from 'ready' to this joint state, in deg (joints 1-6). 
# Supply an optional final joint state here. It is used by both coffee_demo.py and
# main_scene.py. Set to None to end the sequence at 'ready'.
# A goal outside the limits stops the cell with a fault. A goal whose direct
# path would collide is re-planned through a waypoint
FINAL_Q_DEG = [60, -40, -100, 0, 50, 0]



class CoffeeJob:
    """The JAKA coffee station's job for the shared task sequencer."""

    def __init__(self, station, tray):
        self.station = station
        self.tray = tray
        # Conveyor stop for the JAKA. The tray stops with the cup slot exactly
        # at the planner's target at 'on_tray' (r, phi) in the JAKA base frame
        r, phi, _ = POSES["on_tray"]
        slot_x = (station.base * SE3(r * np.cos(phi), r * np.sin(phi), 0)).t[0]
        self.tray_stop_x = slot_x - Tray.CUP_SLOT.t[0]

    def segments(self):
        S = Segment
        segs = [
            S("jtraj", "home -> ready", goal="ready"),
            S("dispense", "dispense a cup"),
            S("jtraj", "ready -> in front of cup", goal="pre_pick"),
            S("rmrc", "approach cup (straight line)", goal="pick", mode="line"),
            S("grip", "close gripper", close=True, hold=True),
            S("rmrc", "lift cup off pad", goal="lift", mode="line"),
            S("jtraj", "empty cup -> coffee machine", goal="pre_insert"),
            S("rmrc", "insert cup into bay", goal="insert", mode="line"),
            S("brew", "brew coffee", frames=int(3.0 / DT)),
            S("rmrc", "withdraw full cup", goal="pre_insert", mode="line"),
            S("rmrc", "carry coffee to tray (level arc)", goal="above_tray", mode="arc",
              requires="tray_at_jaka"),
            S("rmrc", "lower cup onto tray", goal="on_tray", mode="line"),
            S("grip", "release cup", close=False, hold=False),
            S("rmrc", "slide fingers off cup", goal="retreat", mode="line", sets="coffee_placed"),
            S("jtraj", "return to ready", goal="ready"),
        ]
        if FINAL_Q_DEG is not None: # set at the top of this file
            segs.append(S("jtraj", "final joint state (supplied)", goal=np.deg2rad(FINAL_Q_DEG)))
        return segs

    # Targets
    def target_pose(self, name):
        return self.station.tcp_pose(name)

    def target_polar(self, name):
        return POSES[name]

    def ik_seeds(self, name):
        """IK_SEEDS with joint 1 turned to face the target's bearing."""
        seeds = []
        for s in IK_SEEDS:
            s = s.copy()
            s[0] = POSES[name][1]
            seeds.append(s)
        return seeds

    # Non motion segments
    def run_action(self, seg, lane):
        if seg.kind == "grip":
            target = Gripper.CLOSED if seg.p["close"] else Gripper.OPEN
            g = self.station.gripper
            step = (Gripper.OPEN - Gripper.CLOSED) / 10
            g.opening = target if abs(g.opening - target) <= step else g.opening + np.sign(target - g.opening) * step
            finished = g.opening == target
            if finished and not seg.p["close"]:
                # Hand the cup to the tray at its exact current pose.
                cup = self.station.cup
                self.tray.attach(cup, self.tray.pose.inv() * cup.pose)
            return finished
        if seg.kind == "brew":
            if seg.i == 0:
                self.station.set_brewing(True)
            seg.i += 1
            finished = seg.i >= seg.p["frames"]
            if finished:
                self.station.set_brewing(False, done=True)
            return finished
        if seg.kind == "dispense":
            self.station.cup.set_pose(self.station.cup_dispense_pose)
            return True
        return True

    def update_visuals(self, lane):
        """Gripper and cup being held follow the flange every frame"""
        T_flange = lane.robot.fkine(lane.robot.q)
        self.station.gripper.update(T_flange)
        if lane.holding:
            self.station.cup.held_by(T_flange * TOOL)


class CoffeeTask(TaskSequencer):
    """The JAKA station by itself.
    JAKA lane + conveyor lane (coffee_demo.py).
    """

    def __init__(self, robot, planner, station, tray, safety, log=print):
        self.tray = tray
        self.job = CoffeeJob(station, tray)
        self.tray_stop_x = self.job.tray_stop_x
        S = Segment
        conveyor = [
            S("convey", "tray -> JAKA station", x=self.tray_stop_x, sets="tray_at_jaka"),
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="coffee_placed",
              sets="delivered"),
        ]
        lanes = [RobotLane("JAKA", robot, planner, self.job, self.job.segments()),
                 ConveyorLane("conveyor", tray, conveyor)]
        super().__init__(safety, lanes, log)

    @property
    def holding(self):
        return self.lanes[0].holding

    def dry_run(self, lane=None):
        return super().dry_run(lane or self.lanes[0])
