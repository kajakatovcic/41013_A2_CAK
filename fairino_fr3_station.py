"""
FAIRINO FR3 WORKSTATION
=======================

Defines the station associated with the FAIRINO FR3 in the shared catering
workcell.

The station contains:
    - FR3 gripper visual
    - active cold-drink can
    - drink stock
    - workbench
    - movable drink cooler used for collision testing
    - task-space target positions
    - inverse-kinematics seed configurations

--------------
The movable cooler provides the FR3's deliberate upcoming-collision test.

When the cooler is toggled into the carry path:
    1. its visual pose changes
    2. its collision model moves to the same pose
    3. the shared sequencer is notified that the environment changed
    4. the remaining robot path is collision checked
    5. the planner generates an avoidance trajectory if one is available
    6. otherwise the shared safety controller enters a fault state

"""

from math import pi
import numpy as np

from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Mesh

from ir_support_extra_parts import part_path

from collisions import Obstacle
from fairino_fr3 import FairinoFR3

from workcell_layout import (STATIONS, Z_COUNTER)

# -----------------------------------------------------------------------------#
# Gripper
# -----------------------------------------------------------------------------#

class FR3Gripper:

    """
    Simplified two-finger gripper visual.

    The gripper does not require an additional articulated robot model.
    Instead, its palm and fingers are transformed relative to the calculated
    FR3 flange pose every simulation frame.
    """

    PALM = (0.065, 0.080, 0.030)
    FINGER = (0.018, 0.012, 0.090)

    # Distance from TCP centreline to inside face of each finger
    OPEN = 0.055
    CLOSED = 0.034

    # TCP is approximately in the middle of the gripping section
    TOOL_LENGTH = PALM[2] + FINGER[2] / 2

    # Fixed flange-to-gripper mounting transformation.
    # Ry(pi) gives the validated downward-facing gripper orientation.
    GRIPPER_MOUNT = (SE3.Tz(-0.200) * SE3.Ry(pi))

    def __init__(self):

        self.palm = Cuboid(scale=list(self.PALM), color=(0.25, 0.27, 0.30, 1.0))

        self.fingers = [
            Cuboid(scale=list(self.FINGER), color=(0.70, 0.72, 0.75, 1.0)),
            Cuboid(scale=list(self.FINGER), color=(0.70, 0.72, 0.75, 1.0))
        ]

        self.opening = self.OPEN

    def update(self, T_flange):
        """
        Update the gripper visual from the current FR3 flange transform.
        """
        T_mount = T_flange * self.GRIPPER_MOUNT

        self.palm.T = (T_mount * SE3(0, 0, self.PALM[2] / 2)).A

        # Both fingers move symmetrically.
        z = self.PALM[2] + self.FINGER[2] / 2
        y = self.opening + self.FINGER[1] / 2

        self.fingers[0].T = (T_mount * SE3(0, y, z)).A
        self.fingers[1].T = (T_mount * SE3(0, -y, z)).A

    def add_to_env(self, env):

        env.add(self.palm)

        for finger in self.fingers:
            env.add(finger)

# -----------------------------------------------------------------------------#
# Cold drink model
# -----------------------------------------------------------------------------#

class ColdDrink:

    """
    Cylindrical drink representation.

    The same geometry is used for:
        - the active can manipulated by the FR3
        - decorative workstation stock

    Only self.drink in FR3Station participates in the robot task.
    """

    RADIUS = 0.033
    HEIGHT = 0.115

    def __init__(self, pose, color=(0.75, 0.15, 0.15, 1.0)):

        self.shape = Cylinder(radius=self.RADIUS, length=self.HEIGHT, color=color)
        self.set_pose(pose)

    def set_pose(self, pose):

        self.pose = pose
        self.shape.T = pose.A

    def held_by(self, T_tcp, T_drink_in_tcp):

        """
        Maintain the drink's recorded pose relative to the TCP.

        Recording this transform when the gripper closes prevents the object
        from visually snapping or teleporting to an assumed grasp position.
        """
        self.set_pose(T_tcp * T_drink_in_tcp)

    def add_to_env(self, env):
        env.add(self.shape)

# =============================================================================
# Deliberate collision obstacle - drink cooler
# =============================================================================

# Visual dimensions and collision dimensions intentionally match.

# This means the collision volume demonstrated corresponds to the object seen in Swift.

OBSTACLE_CRATE_SIZE = (0.20, 0.10, 0.19)
OBSTACLE_CRATE_COLOUR = (0.15, 0.35, 0.85, 1.0)

# Polar positions relative to the FR3 base.

# PARKED:
# safely outside the normal drink-carry path.

# IN_PATH:
# deliberately placed underneath the normal FR3 carry trajectory so the
# upcoming collision must be detected before the robot reaches it.

OBSTACLE_CRATE_PARKED = (0.35, np.deg2rad(-90))
OBSTACLE_CRATE_IN_PATH = (0.34, np.deg2rad(25))

class ObstacleCrate:
    """
    Visual representation of the blue drink cooler.

    The white strip is visual decoration only. Collision checking uses the
    main cooler dimensions.
    """

    def __init__(self, pose):

        self.shape = Cuboid(scale=list(OBSTACLE_CRATE_SIZE), color=OBSTACLE_CRATE_COLOUR)

        # White label strip on cooler front face
        self.label = Cuboid(scale=[OBSTACLE_CRATE_SIZE[0] * 0.90, 0.006, OBSTACLE_CRATE_SIZE[2] * 0.28], color=(0.95, 0.95, 0.95, 1.0))
        self.set_pose(pose)

    def set_pose(self, pose):

        """
        Move the cooler and visual label as one object.
        """
        self.pose = pose
        self.shape.T = pose.A
        # White label on the front face
        self.label.T = (pose * SE3(0, + (OBSTACLE_CRATE_SIZE[1] / 2 + 0.004), 0.02)).A

# =============================================================================
# FR3 task-space positions
# =============================================================================

# Task positions are expressed using polar coordinates relative to the robot:
#
#     r   = radial distance from FR3 base
#     phi = bearing around FR3 base
#     h   = TCP height
#
# This representation is particularly useful for the drink carry because the
# RMRC arc can move around the robot while preserving the drink orientation.

R_PICK = 0.28
PHI_PICK = 0

R_TRAY = 0.540
PHI_TRAY = np.deg2rad(90)

# Grasp centre is based on the centre of the can plus half the finger length.
H_PICK = (ColdDrink.HEIGHT / 2 + FR3Gripper.FINGER[2] / 2)

# Vertical clearances.
H_ABOVE_PICK = H_PICK + 0.16
H_PLACE = H_PICK + 0.010
H_LIFT = H_PICK + 0.12
H_CLEAR_PLACE = H_PLACE + 0.12


POSES = {

    "above_pick": (R_PICK,PHI_PICK,H_ABOVE_PICK),
    "pick": (R_PICK, PHI_PICK, H_PICK),
    "above_tray": (R_TRAY,PHI_TRAY,H_LIFT),
    "place": (R_TRAY,PHI_TRAY,H_PLACE),
    "clear_tray": (R_TRAY,PHI_TRAY,H_LIFT),
}

# =============================================================================
# Inverse-kinematics seeds
# =============================================================================

# IK is not guaranteed to find the same solution from every initial guess.

# Multiple seeds provide different starting configurations for ikine_LM,
# increasing the chance of finding a valid solution inside the FR3 joint
# limits.

IK_SEEDS = [
    np.deg2rad([0, -90, 0, -90, 0, 0]),
    np.deg2rad([20, -80, 20, -70, 20, 10]),
    np.deg2rad([-20, -70, 30, -60, -20, -10]),
]

# Known useful initial guess for the tray-side solution branch.
TRAY_IK_SEED = np.deg2rad([-63.6, 18.3, -23.8, -174.5, -101.4, 90.0])

# -----------------------------------------------------------------------------#
# FR3 station
# -----------------------------------------------------------------------------#

class FR3Station:

    """
    FAIRINO FR3 station within the shared catering workcell.

    station_base:
        fixed workstation reference from workcell_layout.py

    robot_base:
        actual mounting frame used for FR3 kinematics

    Separating these two allows the robot to be physically offset on the bench
    without changing the whole workstation position.
    """

    def __init__(self, station_base, robot_base):

        # Fixed reference for the physical workstation
        self.station_base = station_base

        # Actual FR3 mounting pose used for robot kinematics
        self.robot_base = robot_base

        # Keep self.base as the robot base so the existing task code
        # continues to use the correct kinematic reference.
        self.base = robot_base

        # Swift sim objects
        self.shapes = []
        # Collision bodies for the planner
        self.obstacles = []

        # Bench stays fixed relative to the workstation
        self._build_bench()
        self._build_obstacle_crate()

        self.gripper = FR3Gripper()

        # ------------------------------------------------------------------
        # Pick up drink
        # ------------------------------------------------------------------
        r = R_PICK
        phi = PHI_PICK

        drink_pose = (self.robot_base * SE3(r * np.cos(phi), r * np.sin(phi), ColdDrink.HEIGHT / 2))
        
        # This is the only can manipulated by the FR3.
        self.drink = ColdDrink(drink_pose)
        # Drink stock on station 
        self.decor_cans = []
        self._build_decor_cans()

    def _obstacle_crate_pose(self, r_phi):
        """
        Convert cooler polar coordinates into a full world transform.
        """
        r, phi = r_phi
        return (self.robot_base * SE3(r * np.cos(phi), r * np.sin(phi), OBSTACLE_CRATE_SIZE[2] / 2))

    def _make_prop_can(self, x, y, color):

        pose = (self.robot_base * SE3(x, y, ColdDrink.HEIGHT / 2))
        return ColdDrink(pose, color=color)

    def _build_decor_cans(self):

        """
        Create evenly spaced red, pink and green drink-stock rows.

        These objects are deliberately NOT inserted into self.obstacles
        because they are background visualisation only.

        This prevents purely decorative objects from changing the validated
        FR3 trajectory or collision behaviour.
        """

        active_x = R_PICK * np.cos(PHI_PICK)
        active_y = R_PICK * np.sin(PHI_PICK)

        # Even spacing between cans
        dx = 0.080
        dy = 0.085

        red = (0.75, 0.15, 0.15, 1.0)
        pink = (0.95, 0.40, 0.90, 1.0)
        green = (0.45, 0.95, 0.55, 1.0)

        # Bottom row - red
        for i in range(5):
            self.decor_cans.append(self._make_prop_can(active_x + (i+1) * dx, active_y, red))

        # Middle row - pink
        for i in range(5):
            self.decor_cans.append(self._make_prop_can(active_x + (i+1) * dx, active_y - dy, pink))

        # Top row - green
        for i in range(5):
            self.decor_cans.append(self._make_prop_can(active_x + (i+1) * dx, active_y - 2 * dy, green))
    
    # ----------------------------------------------------------------------
    # Workbench
    # ----------------------------------------------------------------------

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

        self.bench = Mesh(str(part_path("Workbench")), scale=[0.64, 1.5, 1.0], pose=bench_centre)
        self.shapes.append(self.bench)

    # ----------------------------------------------------------------------
    # Collision obstacle
    # ----------------------------------------------------------------------
    def _build_obstacle_crate(self):

        """
        Create both versions of the drink cooler:

            visual representation -> shown in Swift

            collision representation -> checked by the planner

        The dimensions match so the safety behaviour corresponds to what is
        visible during the demonstration.
        """
        # Start safely parked.
        pose = self._obstacle_crate_pose(OBSTACLE_CRATE_PARKED)

        # Visible blue crate.
        self.obstacle_crate_visual = ObstacleCrate(pose)

        # Matching collision body.
        self.obstacle_crate = Obstacle("FR3 supply crate", list(OBSTACLE_CRATE_SIZE), pose, movable=True)

        self.obstacles.append(self.obstacle_crate)
        self.shapes.append(self.obstacle_crate_visual.shape)
        self.shapes.append(self.obstacle_crate_visual.label)

        self.obstacle_crate_placed = False

    def toggle_obstacle_crate(self):
        """
        Deliberately move the cooler into or out of the FR3 carry path.

        This function itself moves both representations:
            - visible cooler
            - collision obstacle

        The GUI obstacle callback then informs TaskSequencer that the
        environment changed.

        The sequencer checks the REMAINING planned path before the robot
        reaches the cooler. If a collision is predicted, the planner attempts
        a new collision-free trajectory.

        This provides the viva collision-response demonstration.
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

    # Swift visualisation
    def add_to_env(self, env):

        for shape in self.shapes:
            env.add(shape)

        for can in self.decor_cans:
            can.add_to_env(env)

        self.gripper.add_to_env(env)
        self.drink.add_to_env(env)

    # Inverse-kinematics seed selection
    def ik_seeds(self, name):
        """
        Generate multiple starting guesses for inverse kinematics.

        A six-axis arm may have several valid joint-space solutions for the
        same TCP pose.

        Different seeds let ikine_LM search different solution branches rather
        than relying on a single initial guess.
        """
        phi = POSES[name][1]
        seeds = []

        for seed in IK_SEEDS:

            # Solution branch approximately facing the target.
            q1 = seed.copy()
            q1[0] = phi
            seeds.append(q1)

            # Alternate base/shoulder branch.
            q2 = seed.copy()
            q2[0] = phi - pi
            seeds.append(q2)

        # Tray-side tasks also receive a validated seed near the known
        # reachable conveyor configuration.
        if name in ("above_tray", "place", "clear_tray"):
            seeds.append(TRAY_IK_SEED.copy())

        return seeds

# -----------------------------------------------------------------------------
# Build FAIRINO FR3 station and planner
# -----------------------------------------------------------------------------

# Minimum allowable link height relative to the workcell counter.

# This gives the planner another simple workcell constraint so a mathematically
# valid trajectory cannot pass robot links unrealistically through the bench.
MIN_LINK_Z = Z_COUNTER + 0.03


def build_fr3(obstacles=None):
    """
    Construct the complete FR3 subsystem.

    Returns:
        robot
            parameter-based FAIRINO FR3 model

        station
            workstation visuals, targets and obstacles

        planner
            shared MotionPlanner configured using FR3 settings

    The FR3Planner import is intentionally local because fairino_fr3_motion
    requires FR3Gripper and ColdDrink from this module. Importing FR3Planner at
    the top of the file would therefore create a circular import.
    """

    from fairino_fr3_motion import FR3Planner
    
    station_base = STATIONS["FAIRINO_FR3"][0]

    # Validated physical offset between workstation and robot base.
    robot_base = station_base * SE3(0, 0.15, 0)

    robot = FairinoFR3(base=robot_base)
    station = FR3Station(station_base, robot_base)
    planner = FR3Planner(robot, station.obstacles if obstacles is None else obstacles, MIN_LINK_Z)

    return robot, station, planner