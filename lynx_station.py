import numpy as np
from spatialmath import SE3
from spatialgeometry import Mesh, Cylinder, Cuboid
from ir_support_extra_parts import part_path
 
from lynxmotion_ses_pro_model import LynxmotionSESPro, BASE_Z, TOOL_Z
from motion_planner import MotionPlanner, PlannerSettings
from workcell_layout import STATIONS, Z_COUNTER
from math import pi

MIN_LINK_Z = Z_COUNTER + 0.02

PHI_COOKIES = pi           
R_COOKIES = 0.52           
BOARD_SIZE = (0.19, 0.26, 0.012)   
BOARD_T = BOARD_SIZE[2]
 
COOKIE_RADIUS = 0.03
COOKIE_T = 0.010
COOKIE_COLOUR = (0.72, 0.50, 0.25, 1.0)

COOKIE_GRID = [(-0.045, -0.08), (-0.045, 0.0), (-0.045, 0.08),
               (0.045, -0.08), (0.045, 0.0), (0.045, 0.08)]
 
class Cookie:
 
    def __init__(self, pose):
        self.shape = Cylinder(radius=COOKIE_RADIUS, length=COOKIE_T, color=COOKIE_COLOUR)
        self.set_pose(pose)
 
    def set_pose(self, pose):
        self.pose = pose
        self.shape.T = pose.A

class LynxStation:
    
    def __init__(self, robot):
        self.robot = robot
        self.base = STATIONS["Lynxmotion"][0]
        self.shapes = []
        self.obstacles = []
        self.cookies = []

        self._build_bench()
        self._build_cookie_tray()
        self.base_plate, self.parts = robot.link_visuals()
        self._widen_gripper(COOKIE_RADIUS + 0.002)
        robot.pose_visuals(robot.q, self.parts)

    def _build_bench(self):     
        bx,by = self.base.t[0], self.base.t[1]
        self.bench = Mesh(str(part_path("Workbench")), scale=[0.64, 1.5, 1.0],
                          pose=SE3(bx + 0.05, by - 0.1875, 0))
        self.shapes.append(self.bench)

    def _build_cookie_tray(self):
        self.T_cookie_tray = self.base * SE3.Rz(PHI_COOKIES) * SE3(R_COOKIES, 0, 0)
        board = Cuboid(scale=list(BOARD_SIZE), color=(0.55, 0.40, 0.25, 1.0),
                       pose=self.T_cookie_tray * SE3(0, 0, BOARD_T / 2))
        self.shapes.append(board)
        for dx, dy in COOKIE_GRID:
            cookie = Cookie(self.T_cookie_tray * SE3(dx, dy, BOARD_T + COOKIE_T / 2))
            self.cookies.append(cookie)
            self.shapes.append(cookie.shape)
 
    def _widen_gripper(self, half_gap):
        for k, (idx, shape, off) in enumerate(self.parts):
            if idx == 5 and isinstance(shape, Cuboid) and abs(off.t[0]) > 1e-6:
                xc = float(np.sign(off.t[0])) * (half_gap + 0.006)
                self.parts[k] = (idx, shape, SE3(xc, off.t[1], off.t[2]))
 
    def cookie_rest_pose(self, k):
        dx, dy = COOKIE_GRID[k]
        return self.T_cookie_tray * SE3(dx, dy, BOARD_T + COOKIE_T / 2)

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
    
    robot.tool = SE3()
    settings = PlannerSettings(tool=SE3.Tz(TOOL_Z), roll=0.0, pitch=0.0, tcp_speed=0.10,
                               manip_epsilon=0.005, via_min_radius=0.25)
    planner = MotionPlanner(robot, station.obstacles if obstacles is None else obstacles,
                            MIN_LINK_Z, settings)
    return robot, station, planner