import numpy as np

from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh
from ir_support_extra_parts import part_path

from collisions import Obstacle

from math import pi

from fairino_fr3 import FairinoFR3

from workcell_layout import (
    STATIONS,
    Z_COUNTER
)


# -----------------------------------------------------------------------------#
# Gripper
# -----------------------------------------------------------------------------#

class FR3Gripper:

    # Simple visual dimensions in metres
    PALM = (0.065, 0.080, 0.030)
    FINGER = (0.018, 0.012, 0.090)

    # Distance from TCP centreline to inside face of each finger
    OPEN = 0.055
    CLOSED = 0.034

    # TCP is approximately in the middle of the gripping section
    TOOL_LENGTH = PALM[2] + FINGER[2] / 2

    GRIPPER_MOUNT = (SE3.Tz(-0.200) * SE3.Ry(pi))

    def __init__(self):

        self.palm = Cuboid(
            scale=list(self.PALM),
            color=(0.25, 0.27, 0.30, 1.0)
        )

        self.fingers = [
            Cuboid(
                scale=list(self.FINGER),
                color=(0.70, 0.72, 0.75, 1.0)
            ),
            Cuboid(
                scale=list(self.FINGER),
                color=(0.70, 0.72, 0.75, 1.0)
            )
        ]

        self.opening = self.OPEN

    def update(self, T_flange):

        T_mount = T_flange * self.GRIPPER_MOUNT

        self.palm.T = (
            T_mount *
            SE3(0, 0, self.PALM[2] / 2)
        ).A

        z = self.PALM[2] + self.FINGER[2] / 2
        y = self.opening + self.FINGER[1] / 2

        self.fingers[0].T = (
            T_mount *
            SE3(0, y, z)
        ).A

        self.fingers[1].T = (
            T_mount *
            SE3(0, -y, z)
        ).A

    def add_to_env(self, env):

        env.add(self.palm)

        for finger in self.fingers:
            env.add(finger)

# -----------------------------------------------------------------------------#
# Cold drink
# -----------------------------------------------------------------------------#

class ColdDrink:

    RADIUS = 0.033
    HEIGHT = 0.115

    def __init__(self, pose):

        self.shape = Cylinder(
            radius=self.RADIUS,
            length=self.HEIGHT,
            color=(0.75, 0.15, 0.15, 1.0)
        )

        self.set_pose(pose)

    def set_pose(self, pose):

        self.pose = pose
        self.shape.T = pose.A

    def held_by(self, T_tcp, T_drink_in_tcp):

        self.set_pose(
            T_tcp * T_drink_in_tcp
        )

    def add_to_env(self, env):

        env.add(self.shape)

# -----------------------------------------------------------------------------#
# Deliberate movable obstacle crate
# -----------------------------------------------------------------------------#

OBSTACLE_CRATE_SIZE = (
    0.12,   # X width
    0.10,   # Y depth
    0.19    # Z height
)

OBSTACLE_CRATE_COLOUR = (0.15, 0.35, 0.85, 1.0)

# Positions are relative to the FR3 robot base.
# Parked position:
# safely behind / to the staff side of the robot and outside the
# normal pick-to-tray motion.
OBSTACLE_CRATE_PARKED = (
    0.35,                   # r
    np.deg2rad(-90)         # phi
)

# In-path position:
# approximately midway around the FR3's carry arc between:
#   pickup  = 0 deg
#   tray    = 90 deg
#
# This is an initial value and may need small tuning after we see
# the actual collision/replanning behaviour.
OBSTACLE_CRATE_IN_PATH = (
    0.34,                   # r
    np.deg2rad(25)          # phi
)

class ObstacleCrate:
    """
    Visual movable obstacle used to demonstrate FR3 collision
    prediction and replanning.
    """

    def __init__(self, pose):

        self.shape = Cuboid(
            scale=list(OBSTACLE_CRATE_SIZE),
            color=OBSTACLE_CRATE_COLOUR
        )

        self.set_pose(pose)

    def set_pose(self, pose):

        self.pose = pose
        self.shape.T = pose.A

# -----------------------------------------------------------------------------#
# FR3 task target positions
# Stored as (r, phi, h) relative to the FR3 base frame
# -----------------------------------------------------------------------------#

# Physical centre of the can on the bench
#R_CAN = 0.28

# Radial position of the red drink pickup
R_PICK = 0.28

# Safe approach point, 40 mm before the grasp TCP
R_CLEAR_PICK = 0.24

PHI_PICK = 0

R_TRAY = 0.550
PHI_TRAY = np.deg2rad(90)

H_PICK = (ColdDrink.HEIGHT / 2 + FR3Gripper.FINGER[2] / 2)
H_ABOVE_PICK = H_PICK + 0.16
H_PLACE = H_PICK + 0.010
H_LIFT = H_PICK + 0.12
H_MID_PLACE = (H_LIFT + H_PLACE) / 2
H_READY = 0.25
H_CLEAR_PLACE = H_PLACE + 0.12


POSES = {

    "above_pick":
        (R_PICK,PHI_PICK,H_ABOVE_PICK),

    "pick":
        (R_PICK, PHI_PICK, H_PICK),

    "above_tray":
        (R_TRAY,PHI_TRAY,H_LIFT),

    "place":
        (R_TRAY,PHI_TRAY,H_PLACE),

    "clear_tray":
        (R_TRAY,PHI_TRAY,H_LIFT),
}

IK_SEEDS = [

    np.deg2rad([0, -90, 0, -90, 0, 0]),

    np.deg2rad([20, -80, 20, -70, 20, 10]),

    np.deg2rad([-20, -70, 30, -60, -20, -10]),

]

TRAY_IK_SEED = np.deg2rad([
    -63.6,
     18.3,
    -23.8,
   -174.5,
   -101.4,
     90.0
])

# -----------------------------------------------------------------------------#
# FR3 station
# -----------------------------------------------------------------------------#

class FR3Station:

    def __init__(self, station_base, robot_base):

        # Fixed reference for the physical workstation
        self.station_base = station_base

        # Actual FR3 mounting pose used for robot kinematics
        self.robot_base = robot_base

        # Keep self.base as the robot base so the existing task code
        # continues to use the correct kinematic reference.
        self.base = robot_base

        self.shapes = []
        self.obstacles = []

        # Bench stays fixed relative to the workstation
        self._build_bench()
        self._build_obstacle_crate()

        self.gripper = FR3Gripper()

        # Initial drink remains a workstation prop for now
        r = R_PICK
        phi = PHI_PICK

        drink_pose = (
            self.robot_base
            * SE3(
                r * np.cos(phi),
                r * np.sin(phi),
                ColdDrink.HEIGHT / 2
            )
        )

        self.drink = ColdDrink(drink_pose)

    def _obstacle_crate_pose(self, r_phi):

        r, phi = r_phi

        return (
            self.robot_base
            * SE3(
                r * np.cos(phi),
                r * np.sin(phi),
                OBSTACLE_CRATE_SIZE[2] / 2
            )
        )
    
    def _build_bench(self):

        bx = self.station_base.t[0]
        by = self.station_base.t[1]

        # Same Workbench asset/style used by the other robot stations.
        # Most of the bench sits on the staff side of the robot.
        bench_centre = SE3(
            bx + 0.05,
            by - 0.1875,
            0
        )

        self.bench = Mesh(
            str(part_path("Workbench")),
            scale=[0.64, 1.5, 1.0],
            pose=bench_centre
        )

        self.shapes.append(self.bench)

    def _build_obstacle_crate(self):

        # Start safely parked.
        pose = self._obstacle_crate_pose(
            OBSTACLE_CRATE_PARKED
        )

        # Visible blue crate.
        self.obstacle_crate_visual = ObstacleCrate(pose)

        # Matching collision body.
        self.obstacle_crate = Obstacle(
            "FR3 supply crate",
            list(OBSTACLE_CRATE_SIZE),
            pose,
            movable=True
        )

        self.obstacles.append(self.obstacle_crate)
        self.shapes.append(self.obstacle_crate_visual.shape)

        self.obstacle_crate_placed = False

    def toggle_obstacle_crate(self):
        """
        Move the blue crate into or out of the FR3 carry path.

        Returns True when the obstacle is now in the path.
        """

        self.obstacle_crate_placed = not self.obstacle_crate_placed

        target = (
            OBSTACLE_CRATE_IN_PATH
            if self.obstacle_crate_placed
            else OBSTACLE_CRATE_PARKED
        )

        pose = self._obstacle_crate_pose(target)

        # Move visible crate.
        self.obstacle_crate_visual.set_pose(pose)

        # Move collision body.
        self.obstacle_crate.move_to(pose)

        return self.obstacle_crate_placed

    def add_to_env(self, env):

        for shape in self.shapes:
            env.add(shape)

        self.gripper.add_to_env(env)
        self.drink.add_to_env(env)

    def ik_seeds(self, name):

        phi = POSES[name][1]

        seeds = []

        for seed in IK_SEEDS:

            # Normal bearing branch
            q1 = seed.copy()
            q1[0] = phi
            seeds.append(q1)

            # Alternate base/shoulder branch
            q2 = seed.copy()
            q2[0] = phi - pi
            seeds.append(q2)

        # Known reachable branch for the conveyor/tray side
        if name in ("above_tray", "place", "clear_tray"):
            seeds.append(TRAY_IK_SEED.copy())

        return seeds

# -----------------------------------------------------------------------------
# Build FAIRINO FR3 station
# -----------------------------------------------------------------------------

MIN_LINK_Z = Z_COUNTER + 0.03


def build_fr3(obstacles=None):
    """
    Build the FAIRINO FR3, its station and its planner.

    Used by both the standalone FR3 demo and the integrated main scene.
    """
    from fairino_fr3_motion import FR3Planner
    
    station_base = STATIONS["FAIRINO_FR3"][0]

    # Validated physical offset between workstation and robot base.
    robot_base = station_base * SE3(0, 0.15, 0)

    robot = FairinoFR3(base=robot_base)

    station = FR3Station(
        station_base,
        robot_base
    )

    planner = FR3Planner(
        robot,
        station.obstacles if obstacles is None else obstacles,
        MIN_LINK_Z
    )

    return robot, station, planner