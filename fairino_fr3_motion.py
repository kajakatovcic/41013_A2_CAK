from math import pi

from spatialmath import SE3

from motion_planner import MotionPlanner, PlannerSettings
from fairino_fr3_station import FR3Gripper

# -----------------------------------------------------------------------------#
# Tool / TCP definition
# -----------------------------------------------------------------------------#

# The shared MotionPlanner expects TOOL to be a PURE TRANSLATION
# from the mathematical flange frame to the TCP.
#
# Physical gripper mount:
#   -0.200 m from flange
# Gripping centre:
#   another 0.075 m outward
#
# Total TCP offset:
TOOL_OFFSET = 0.200 + FR3Gripper.TOOL_LENGTH

TOOL = SE3.Tz(-TOOL_OFFSET)


# -----------------------------------------------------------------------------#
# Grasp orientation
# -----------------------------------------------------------------------------#

# Starter orientation for a side grasp.
# We may tune these values after checking IK.
GRASP_ROLL = 0.0
GRASP_PITCH = 0.0


# -----------------------------------------------------------------------------#
# Motion planner settings
# -----------------------------------------------------------------------------#

TCP_SPEED = 0.15


FR3_SETTINGS = PlannerSettings(

    tool=TOOL,

    roll=GRASP_ROLL,
    pitch=GRASP_PITCH,

    # With pitch = pi/2, tool +X points downward,
    # therefore -X points upward through the drink.
    payload_up=(0, 0, 1),

    tcp_speed=TCP_SPEED,

    manip_epsilon=0.005,
    via_min_radius=0.20,

    tool_chain=[TOOL],
    payload_chain=[],

    tool_ellipsoids=[],
    payload_ellipsoids=[],
)


# -----------------------------------------------------------------------------#
# FR3 motion planner
# -----------------------------------------------------------------------------#

class FR3Planner(MotionPlanner):

    def __init__(
        self,
        robot,
        obstacles,
        min_link_z
    ):

        super().__init__(
            robot,
            obstacles,
            min_link_z,
            FR3_SETTINGS
        )