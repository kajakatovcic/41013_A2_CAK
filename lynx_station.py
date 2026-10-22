import numpy as np
from spatialmath import SE3
from spatialgeometry import Mesh, Cylinder, Cuboid
from ir_support_extra_parts import part_path

from collisions import Obstacle
 
from lynxmotion_ses_pro_model import LynxmotionSESPro
from lynx_motion import LynxPlanner
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

# Cookie jar: the Lynxmotion's movable obstacle, the same idea as the JAKA's
# milk pitcher: ir_support_extra_parts 'CandyJar' scaled to 65%
# (0.098 x 0.101 x 0.105 m). Full size (0.161 m tall) it is taller than the
# largest RMRC hop can clear, so the cell could only stop for it, not avoid it.
# Parked on the bench behind the robot (out of every path); the GUI toggle
# moves it onto the bench under the cookie's carry arc, which runs from about
# 170 deg to 94 deg round the base at ~0.1 m above it. 145 deg / 0.50 m is on
# that arc while still on the bench and 8 cm clear of the cookie board, so the
# planner must re-plan the carry over it (RMRC sine hop).
JAR_SCALE = 0.65
JAR_SIZE = (0.151 * JAR_SCALE, 0.155 * JAR_SCALE, 0.161 * JAR_SCALE)
JAR_PARKED = (0.45, -pi / 2)      # (r, phi) in the station base frame
JAR_IN_PATH = (0.50, 145 * pi / 180)

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
        self._build_jar()
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
 
    def _jar_pose(self, r_phi):
        r, phi = r_phi
        return self.base * SE3(r * np.cos(phi), r * np.sin(phi), 0)

    def _build_jar(self):
        """
        The cookie jar and its collision box. Like every obstacle in the cell
        it is a Lab 5 RectangularPrism (with a Lab 6 point cloud) the size of
        the mesh and placed by the jar's SE3.
        """
        pose = self._jar_pose(JAR_PARKED)
        self.jar_mesh = Mesh(str(part_path("CandyJar")), scale=[JAR_SCALE] * 3, pose=pose)
        self.jar = Obstacle("cookie jar", JAR_SIZE, pose * SE3(0, 0, JAR_SIZE[2] / 2),
                            movable=True)
        self.obstacles.append(self.jar)
        self.shapes.append(self.jar_mesh)
        self.jar_placed = False

    def toggle_jar(self):
        """Move the cookie jar into/out of the carry path. Returns the new state."""
        self.jar_placed = not self.jar_placed
        pose = self._jar_pose(JAR_IN_PATH if self.jar_placed else JAR_PARKED)
        self.jar_mesh.T = pose.A
        self.jar.move_to(pose * SE3(0, 0, JAR_SIZE[2] / 2))
        return self.jar_placed

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
    
    # The tool offset is applied by the planner (lynx_motion.TOOL) so the
    # model's own tool is cleared to avoid applying it twice.
    robot.tool = SE3()
    planner = LynxPlanner(robot, station.obstacles if obstacles is None else obstacles,
                          MIN_LINK_Z)
    return robot, station, planner