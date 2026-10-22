"""
FAIRINO FR3 MOTION-PLANNER CONFIGURATION
========================================

The workcell uses one common MotionPlanner for all robots.

This file supplies the FR3-specific information required by that planner:
    - flange-to-TCP offset
    - grasp orientation
    - Cartesian speed
    - manipulability threshold
    - gripper collision geometry
    - held-drink collision geometry

--------------
The FR3 deliberately uses different trajectory methods for different parts
of the task:

    jtraj:
        used for free-space joint-space transfers where the exact TCP path
        is not critical.

    RMRC:
        used for controlled Cartesian approaches, lifts, placement and the
        level carry of the drink.

This is a purposeful selection rather than using one trajectory technique
for every movement.
"""

from spatialmath import SE3

from motion_planner import MotionPlanner, PlannerSettings
from fairino_fr3_station import FR3Gripper, ColdDrink

# =============================================================================
# Tool: Tool Centre Point (TCP)
# =============================================================================

# The shared planner expects TOOL to be a pure translation from the robot
# flange to the TCP.

# flange
#   |
#   | 0.200 m visual mounting offset
#   |
# gripper
#   |
#   | distance to gripping centre
#   |
#  TCP

# Orientation is handled separately through roll/pitch/yaw.

TOOL_OFFSET = 0.200 + FR3Gripper.TOOL_LENGTH
TOOL = SE3.Tz(-TOOL_OFFSET)

# =============================================================================
# Grasp orientation
# =============================================================================

# Top-down grasp.
# The planner changes yaw as the robot moves around its base, while roll and
# pitch remain fixed. This allows the drink to remain upright while being
# transported.
GRASP_ROLL = 0.0
GRASP_PITCH = 0.0

# =============================================================================
# Motion constraints
# =============================================================================

# Cartesian speed used during RMRC.
# The value is deliberately moderate for controlled collaborative motion
# rather than maximising simulated cycle speed.

TCP_SPEED = 0.15

# =============================================================================
# Collision representation
# =============================================================================

# COLLISION METHOD 1:
# Ellipsoid surrounding the gripper.
# This gives the collision checker physical volume around the end effector
# rather than modelling it only as an infinitely thin line.

GRIPPER_ELLIPSOID = (SE3.Tz(-0.260), (0.055, 0.090, 0.100))

# When grasped, the centre of the drink is approximately half a finger length below the TCP.
CAN_CENTRE_FROM_TCP = -FR3Gripper.FINGER[2] / 2

def _can_centreline():

    """
    Generate the collision centre-line of the held can.

    COLLISION METHOD 2:
    The shared collision system uses this line between the top and bottom of
    the drink for line-plane intersection checks.

    The payload geometry is only active while the robot lane is marked as
    holding the drink.
    """

    half_height = ColdDrink.HEIGHT / 2

    return [
        TOOL * SE3.Tz(CAN_CENTRE_FROM_TCP + half_height),
        TOOL * SE3.Tz(CAN_CENTRE_FROM_TCP - half_height),
    ]


# The visible drink is approximately:
#     diameter = 66 mm
#     height   = 115 mm

# The collision ellipsoid is deliberately slightly larger than the physical
# cylinder. This gives a conservative collision margin.
CAN_ELLIPSOID = (TOOL * SE3.Tz(CAN_CENTRE_FROM_TCP),(0.050, 0.050, 0.085))

# =============================================================================
# Planner configuration
# =============================================================================

FR3_SETTINGS = PlannerSettings(

    # Flange-to-TCP transformation.
    tool=TOOL,

    # Fixed top-down grasp orientation.
    roll=GRASP_ROLL,
    pitch=GRASP_PITCH,

    # Direction in the TCP frame that must remain upright while carrying.
    payload_up=(0, 0, 1),

    # Cartesian velocity constraint used by RMRC.
    tcp_speed=TCP_SPEED,

    # Manipulability threshold used by the shared RMRC implementation.
    
    # If Yoshikawa manipulability drops below this value, the planner uses
    # damped least squares to reduce instability close to a singularity.
    
    # This is singularity MITIGATION rather than a guarantee that singular
    # configurations can never occur.
    manip_epsilon=0.005,

    # Prevent generated avoidance waypoints being placed unrealistically close to the base axis.
    via_min_radius=0.20,

    # Line-based collision representations.
    tool_chain=[TOOL],
    # Vertical line through the held can only enabled when lane.holding == True
    payload_chain=_can_centreline(),
    # Physical width/volume of the gripper
    tool_ellipsoids=[GRIPPER_ELLIPSOID],
    # Physical width/volume of the held drink only when holding
    payload_ellipsoids=[CAN_ELLIPSOID],
)


# -----------------------------------------------------------------------------#
# FR3 motion planner
# -----------------------------------------------------------------------------#

class FR3Planner(MotionPlanner):
    """
    FR3-configured version of the common MotionPlanner.

    """
    def __init__(self, robot, obstacles, min_link_z):
        super().__init__(robot, obstacles, min_link_z, FR3_SETTINGS)