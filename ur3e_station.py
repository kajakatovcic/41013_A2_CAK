# UR3e plate station: bench, plate dispenser plates and the suction
# cup as its "tool"
#
# Every prop is placed from the robot's BASE frame with the same polar
# convention as the shared planner, (r, phi, h) - r = distance from the base
# axis, phi = bearing from the base x-axis, h = height above the base plate
# (= the counter top). This means a plate sits exactly where the planner reaches for
# it. If you move the UR3e base in workcell_layout.STATIONS,  the
# station follows.
#
# The UR3e base is mounted turned +90 deg about z (workcell_layout.STATIONS),
# so its x-axis points at the conveyor:
#
#                plate dispenser       UR3e
#                phi = +90 deg   ------ o
#                                       |
#                         tray plate slot on the conveyor, phi = 0 deg
#
# The dispenser is on the upper side so the loaded carry to the tray is a
# quarter turn and staff refill plates from the upstream end of the cell
# away from the moving conveyor. (safety)
#
# REACH: the UR3e is a small arm (0.244 + 0.213m links). With the 0.133m
# suction tool pointing down it reaches about 0.50m at counter height but only
# 0.45m at 0.20m up and 0.35m at 0.30m up (measured by an IK sweep). So the
# dispenser and the tray slot sit close to the base and the plate is carried
# low. Short guide posts keep the lift height (and the carry) at 0.19m.

from math import pi

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh
from ir_support_extra_parts import part_path

from collisions import Obstacle
from ir_support import UR3e

from ur3e_motion import suction_pose, UR3ePlanner, ADAPTER_HEIGHT, CUP_SCALE, PLATE_CONTACT
from workcell_layout import STATIONS, Z_COUNTER, CONVEYOR_Y, BELT_WIDTH
from workcell_scene import Tray

# Plate dispenser (in the UR3e base frame)
PHI_DISPENSER = pi / 2
PHI_TRAY = 0.0
R_DISPENSER = 0.35
R_TRAY = 0.42 # = UR3e base to the tray's plate slot (workcell_layout.UR3E_Y)

DISPENSER_BASE = 0.03 # height of the dispenser's base plate
N_PLATES = 6
PLATE_PITCH = 0.010 # nested plates stack 10mm apart
POST_RADIUS = 0.15 # guide posts around the stack (plate rim at 0.115)
POST_HEIGHT = 0.12 # tall enough to guide the 6-plate stack, short enough that
                  # the lift (and carry) height stays inside the UR3e's reach

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
    "ready":       (0.28, pi / 4, 0.25), # between the dispenser and the tray
    "above_stack": (R_DISPENSER, PHI_DISPENSER, H_ABOVE_STACK),
    "pick":        (R_DISPENSER, PHI_DISPENSER, H_PICK),
    "above_tray":  (R_TRAY, PHI_TRAY, H_CARRY),
    "on_tray":     (R_TRAY, PHI_TRAY, H_PLACE),
    "clear_tray":  (R_TRAY, PHI_TRAY, H_CLEAR_TRAY),
}

# Seeds for ikine_LM with the tool pointing down, all on the same elbow branch
# (checked at every target: other seeds flip the UR3e to a folded pose at the
# pick). joint 1 is replaced with the target bearing when used.
IK_SEEDS = [
    np.array([0, -1.2, -1.6, -1.9, pi / 2, 0]),
    np.array([0, -pi / 2, -pi / 2, -pi / 2, pi / 2, 0]),
]

# Named joint state at the "ready" pose (like qready in the labs), solved once
# with ikine_LM from IK_SEEDS[0]. The first move from home goes here in joint
# space: solving the ready pose from the straight-up home pose would pick the
# least-travel elbow-down branch, which drops the shoulder link to 0.05 m above
# the counter at the tray. From here every later IK stays on this branch.
READY_Q = np.deg2rad([72.9, -81.8, -75.7, -112.6, 90.0, -62.1])


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
    """suction tool on the UR3e flange. Built a 'fake' adapter block + suction cup."""

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
    Builds the bench and plate dispenser around the UR3e and owns their
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
        ir_support_extra_parts 'Workbench' scaled to 1.6 x 1.125m. The 0.93m
        top is the counter height. Its front edge is placed against the
        conveyor's side frame (not relative to the base), so moving the base
        cannot push the bench onto the belt.
        """
        bx = self.base.t[0]
        front_y = CONVEYOR_Y - BELT_WIDTH / 2 - 0.02
        self.shapes.append(Mesh(str(part_path("Workbench")), scale=[0.64, 1.5, 1.0],
                                pose=SE3(bx - 0.1, front_y - 1.125 / 2, 0)))

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


# Link frames must stay 5cm above the counter
MIN_LINK_Z = Z_COUNTER + 0.05


def build_ur3e(obstacles=None):
    """The UR3e, its plate station and its planner. The UR3e starts at its
    home pose (arm straight up) as set by the ir_support model."""
    base = STATIONS["UR3e"][0]
    robot = UR3e(base=base)
    station = PlateStation(base)
    planner = UR3ePlanner(robot, station.obstacles if obstacles is None else obstacles, MIN_LINK_Z)
    return robot, station, planner
