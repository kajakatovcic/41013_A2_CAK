import numpy as np
from spatialmath import SE3
from spatialgeometry import Mesh
from ir_support_extra_parts import part_path
 
from lynxmotion_ses_pro_model import LynxmotionSESPro
from motion_planner import MotionPlanner, PlannerSettings
from workcell_layout import STATIONS, Z_COUNTER

MIN_LINK_Z = Z_COUNTER + 0.02

class LynxStation:
    
    def __init__(self, robot):
        self.robot = robot
        self.base = STATIONS["Lynxmotion"][0]
        self.shapes = []
        self.obstacles = []

        self._build_bench()
        self.base_plate, self.parts = robot.link_visuals()
        robot.pose_visuals(robot.q, self.parts)

    def _build_bench(self):
        
        bx,by = self.base.t[0], self.base.t[1]
        self.bench = Mesh(str(part_path("Workbench")), scale=[0.64, 1.5, 1.0],
                          pose=SE3(bx + 0.05, by - 0.1875, 0))
        self.shapes.append(self.bench)

    def update_visuals(self, q):
        self.robot.pose_visuals(q, self.parts)

    def add_to_env(self, env):
        for s in self.shapes:
            env.add(s)
        env.add(self.base_plate)
        for _, shape, _ in self.parts:
            env.add(shape)

def build_lynx(obstacles=None):

    robot = LynxmotionSESPro(mount=STATIONS["Lynxmotion"][0])
    robot.q = np.zeros(6)
    station = LynxStation(robot)
    planner = MotionPlanner(robot, station.obstacles if obstacles is None else obstacles,
                            MIN_LINK_Z, PlannerSettings(tool=SE3()))
    return robot, station, planner