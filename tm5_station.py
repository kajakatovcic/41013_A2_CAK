# OMRON TM5-700 plate station: bench, plate dispenser plates and the suction
# cup as its "tool"
#
# Every prop is placed from the robot's BASE frame with the same polar
# convention as the shared planner, (r, phi, h) - r = distance from the base
# axis, phi = bearing from the base x-axis, h = height above the base plate
# (= the counter top). This means a plate sits exactly where the planner reaches for
# it. If you move the TM5 base in workcell_layout.STATIONS,  the
# station follows.
#
#                plate dispenser       TM5
#                phi = 180 deg   ------ o
#                                       |
#                         tray plate slot on the conveyor, phi = +90 deg
#
# The dispenser is on the upper side so the loaded carry to the tray is a
# quarter turn and staff refill plates from the upstream end of the cell
# away from the moving conveyor. (safety)

from math import pi

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh
from ir_support_extra_parts import part_path

from collisions import Obstacle
from tm5_motion import suction_pose, ADAPTER_HEIGHT, CUP_SCALE, PLATE_CONTACT
from workcell_scene import Tray

# Plate dispenser (in the TM5 base frame)
PHI_DISPENSER = pi
PHI_TRAY = pi / 2
R_DISPENSER = 0.45
R_TRAY = 0.55 # = TM5 base to the tray's plate slot (workcell_layout.TM5_Y)

DISPENSER_BASE = 0.03 # height of the dispenser's base plate
N_PLATES = 6
PLATE_PITCH = 0.010 # nested plates stack 10mm apart
POST_RADIUS = 0.15 # guide posts around the stack (plate rim at 0.115)
POST_HEIGHT = 0.22

# Heights of the suction cup's contact point (TCP) above the counter
H_PICK = DISPENSER_BASE + (N_PLATES - 1) * PLATE_PITCH + PLATE_CONTACT
# Above the stack the plate's collision ellipsoid (0.03m below its mid-height)
# must clear the top of the guide posts.
H_ABOVE_STACK = DISPENSER_BASE + POST_HEIGHT + 0.04
H_PLACE = Tray.FLOOR + PLATE_CONTACT
# The plate is carried level at the lift height and only lowered once it is
# over the tray
H_CARRY = H_ABOVE_STACK
H_CLEAR_TRAY = H_PLACE + 0.115 # empty suction cup lifted clear of the plate

# Named TCP targets as (r, phi, h). Used by the task and by the props below.
POSES = {
    "ready":       (0.35, 3 * pi / 4, 0.30), # between the dispenser and the tray
    "above_stack": (R_DISPENSER, PHI_DISPENSER, H_ABOVE_STACK),
    "pick":        (R_DISPENSER, PHI_DISPENSER, H_PICK),
    "above_tray":  (R_TRAY, PHI_TRAY, H_CARRY),
    "on_tray":     (R_TRAY, PHI_TRAY, H_PLACE),
    "clear_tray":  (R_TRAY, PHI_TRAY, H_CLEAR_TRAY),
}

# Seeds for ikine_LM - elbow up with the tool pointing down. joint 1 is
# replaced with the target bearing when used.
IK_SEEDS = [
    np.array([0, -0.6, 1.6, -1.0, 1.57, 0]),
    np.array([0, 0.3, 1.8, -0.5, 1.57, 0]),
    np.array([0, -0.3, 2.0, -1.7, -1.57, 0]),
]


class Plate:
    """
    'Plate' from ir_support_extra_parts scaled down with frame at the centre of
    its base. While held, its pose follows the TCP. the cup seals on the plate's
    centre (PLATE_CONTACT above the base) and the tool z (down) is flipped back
    to world-up (with SE3.Rx(pi))
    """
    IN_TCP = SE3(0, 0, PLATE_CONTACT) * SE3.Rx(pi)

    def __init__(self, pose):
        self.mesh = Mesh(str(part_path("Plate")), scale=[0.5, 0.5, 0.5])
        self.set_pose(pose)

    def set_pose(self, pose):
        self.pose = pose
        self.mesh.T = pose.A

    def held_by(self, T_tcp):
        self.set_pose(T_tcp * self.IN_TCP)


class SuctionTool:
    """suction tool on the TM5 flange. Built a 'fake' adapter block + suction cup."""

    def __init__(self):
        self.adapter = Cuboid(scale=[0.06, 0.06, ADAPTER_HEIGHT], color=(0.20, 0.22, 0.25, 1.0))
        # The SuctionCup mesh's open lip is at its +z end, so it mounts with
        # its z along the tool z, starting at the adapter's face.
        self.cup = Mesh(str(part_path("SuctionCup")), scale=[CUP_SCALE] * 3,
                        color=(0.15, 0.15, 0.15, 1.0))

    def update(self, T_flange):
        self.adapter.T = (T_flange * SE3(0, 0, ADAPTER_HEIGHT / 2)).A
        self.cup.T = (T_flange * SE3(0, 0, ADAPTER_HEIGHT)).A

    def add_to_env(self, env):
        env.add(self.adapter)
        env.add(self.cup)


class PlateStation:
    """
    Builds the bench and plate dispenser around the TM5 and owns their
    collision obstacles. 'obstacles' is handed to the planner as a list
    """

    def __init__(self, base):
        self.base = base
        self.shapes = []
        self.obstacles = []
        self._build_bench()
        self._build_dispenser()
        self.tool = SuctionTool()

    def _build_bench(self):
        """
        ir_support_extra_parts 'Workbench' scaled. The 0.93m top is the
        counter height
        """
        bx, by = self.base.t[0], self.base.t[1]
        self.shapes.append(Mesh(str(part_path("Workbench")), scale=[0.64, 1.5, 1.0],
                                pose=SE3(bx - 0.1, by - 0.2575, 0)))

    def _build_dispenser(self):
        """
        Plate dispenser built from a base plate, a stack of plates and 3 guide posts.
        Frame is at the stack centre with x pointing away from the robot.
        """
        self.T_dispenser = self.base * SE3.Rz(PHI_DISPENSER) * SE3(R_DISPENSER, 0, 0)
        T = self.T_dispenser
        self.shapes.append(Cylinder(radius=0.16, length=DISPENSER_BASE, color=(0.30, 0.30, 0.32, 1.0),
                                    pose=T * SE3(0, 0, DISPENSER_BASE / 2)))

        # All plates. The top one is the one that is picked
        self.plates = [Plate(T * SE3(0, 0, DISPENSER_BASE + i * PLATE_PITCH))
                       for i in range(N_PLATES)]
        self.shapes += [p.mesh for p in self.plates]

        # Guide posts. Each post's collision box is turned to face the stack.
        # the Lab 5 RectangularPrism is moved by the post's SE3 (collisions.make_obstacle_mesh)
        # so it doesnt need to be aligned with the world axes
        for k in range(3):
            theta = k * 2 * pi / 3
            pose = (T * SE3(POST_RADIUS * np.cos(theta), POST_RADIUS * np.sin(theta),
                            DISPENSER_BASE + POST_HEIGHT / 2) * SE3.Rz(theta))
            self.shapes.append(Cylinder(radius=0.01, length=POST_HEIGHT,
                                        color=(0.75, 0.76, 0.78, 1.0), pose=pose))
            self.obstacles.append(Obstacle(f"plate guide post {k + 1}", [0.02, 0.02, POST_HEIGHT], pose))

    @property
    def top_plate(self):
        return self.plates[-1]

    def tcp_pose(self, name):
        """World TCP pose of a named station target"""
        return suction_pose(self.base, *POSES[name])

    def add_to_env(self, env):
        for s in self.shapes:
            env.add(s)
        self.tool.add_to_env(env)
