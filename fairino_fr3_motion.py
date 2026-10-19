from math import pi

from spatialmath import SE3

from motion_planner import MotionPlanner, PlannerSettings
from fairino_fr3_station import FR3Gripper, ColdDrink

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

# Top-down grasp orientation.
# With roll = pitch = 0, the TCP remains vertical and
# the held drink stays upright along tool +Z.
GRASP_ROLL = 0.0
GRASP_PITCH = 0.0


# -----------------------------------------------------------------------------#
# Motion planner settings
# -----------------------------------------------------------------------------#

TCP_SPEED = 0.15

# -----------------------------------------------------------------------------#
# Collision geometry
# -----------------------------------------------------------------------------#

# The FR3 gripper hangs below the flange.
#
# Visual geometry:
#   mount begins about 0.200 m below the flange
#   fingers extend to about 0.320 m below the flange
#
# The ellipsoid gives the gripper physical width for the
# Lab 6 ellipsoid-vs-point-cloud collision check.
GRIPPER_ELLIPSOID = (SE3.Tz(-0.260), (0.055, 0.090, 0.100))

# The current top-down grasp holds the centre of the drink
# approximately 45 mm below the TCP.
CAN_CENTRE_FROM_TCP = -FR3Gripper.FINGER[2] / 2

def _can_centreline():
    """
    Lab 5 line-plane collision geometry for the held drink.

    Returns the top and bottom points of a centre-line through
    the can. The shared planner connects these as a line segment.
    """

    half_height = ColdDrink.HEIGHT / 2

    return [
        TOOL * SE3.Tz(CAN_CENTRE_FROM_TCP + half_height),
        TOOL * SE3.Tz(CAN_CENTRE_FROM_TCP - half_height),
    ]


# The can is a cylinder:
#   diameter = 66 mm
#   height   = 115 mm
#
# Slightly oversized ellipsoid so the physical can sits inside it.
CAN_ELLIPSOID = (TOOL * SE3.Tz(CAN_CENTRE_FROM_TCP),(0.050, 0.050, 0.085))


FR3_SETTINGS = PlannerSettings(

    tool=TOOL,

    roll=GRASP_ROLL,
    pitch=GRASP_PITCH,

    payload_up=(0, 0, 1),

    tcp_speed=TCP_SPEED,

    manip_epsilon=0.005,
    via_min_radius=0.20,

    #Flange -> TCP line through the gripper
    tool_chain=[TOOL],
    #vertical line through the held can only enabled when lane.holding == True
    payload_chain=_can_centreline(),
    #Physical width/volume of the gripper
    tool_ellipsoids=[GRIPPER_ELLIPSOID],
    #Physical width/volume of the held drink only when holding
    payload_ellipsoids=[CAN_ELLIPSOID],
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