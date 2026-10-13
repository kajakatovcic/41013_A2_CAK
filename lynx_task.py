import numpy as np

from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer
from workcell_layout import STATIONS, COLLECTION_X

DEG = np.pi / 180

FINAL_Q_DEG = None

TRAY_STOP_DX = 0

Q_READY = np.array([0, -35, 75, 0, -40, 0]) * DEG
Q_OVER_TRAY = np.array([90, -35, 75, 0, -40, 0]) * DEG

class LynxJob:
    def __init__(self, station, tray):
        self.station = station
        self.tray = tray
        self.tray_stop_x = STATIONS["Lynxmotion"][0].t[0] + TRAY_STOP_DX

    def segments(self):
        S = Segment
        segs = [
            S("jtraj", "home -> ready", goal=Q_READY),
            S("jtraj", "swing over tray", goal=Q_OVER_TRAY, requires="tray_at_lynx"),
            S("jtraj", "return to ready", goal=Q_READY, sets="food_placed"),
        ]
        if FINAL_Q_DEG is not None:
            segs.append(S("jtraj", "final joint state (supplied)", goal=np.deg2rad(FINAL_Q_DEG)))
        return segs

    def target_pose(self, name):
        raise NotImplementedError("Lynxmotion segments use joint goals for now")
 
    def target_polar(self, name):
        raise NotImplementedError("Lynxmotion has no RMRC targets yet")
 
    def ik_seeds(self, name):
        return []
 
    def run_action(self, seg, lane):
        return True
 
    def update_visuals(self, lane):
        self.station.update_visuals(lane.robot.q)

class LynxTask(TaskSequencer):

    def __init__(self, robot, planner, station, tray, safety, log=print):
        self.tray = tray
        self.job = LynxJob(station, tray)
        S = Segment
        conveyor = [
            S("convey", "tray -> Lynxmotion station", x=self.job.tray_stop_x, sets="tray_at_lynx"),
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="food_placed", sets="delivered"),
        ]
        lanes = [RobotLane("Lynxmotion", robot, planner, self.job, self.job.segments()),
                 ConveyorLane("conveyor", tray, conveyor)]
        super().__init__(safety, lanes, log)
 
    def dry_run(self, lane=None):
        return super().dry_run(lane or self.lanes[0])