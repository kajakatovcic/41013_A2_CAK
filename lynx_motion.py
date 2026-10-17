# Motion settings for the Lynxmotion SES-PRO cookie station.
#
# The planning methods (jtraj, RMRC, IK, collision checks, avoidance) are
# shared by every robot and live in motion_planner.py. This file holds only
# what is specific to the Lynxmotion e.g. its gripper, its tool-down grasp, its
# speed and the collision shapes of the gripper and the cookie
#
# IK uses ikine_LM

from spatialmath import SE3

from lynxmotion_ses_pro_model import TOOL_Z, LEN_L1
from motion_planner import MotionPlanner, PlannerSettings

# Tool- the gripper hangs below the last joint. The model's own tool is set to
# SE3() in build_lynx and the offset is given here instead so it is applied
# once (by the planner)
TOOL = SE3.Tz(TOOL_Z)

# Tool pointing down. With roll = pitch = 0 the flange z-axis points UP and
# the gripper hangs along flange -z (TOOL_Z is negative), so the TCP is below
# the flange and the cookie's up direction is tool +z.
TCP_SPEED = 0.10 # m/s, slow for a light, loose cookie
MANIP_EPSILON = 0.005
VIA_MIN_RADIUS = 0.25

# Cookie (lynx_station.Cookie)- 0.06m across, 0.01m thick held centred on
# the TCP between the fingers.
COOKIE_RADIUS = 0.03
COOKIE_THICKNESS = 0.01

# Collision shapes:
# ELLIPSOIDS used for gripper as a centreline alone cannot tell if the width
# clears an obstacle. an ellipsoid carries width.
# LINE-PLANE used as the obstacle side of the ellipsoid test is a sampled 
# point cloud which can be passed between points.
# the tool centre-line and a cross of lines across the cookie stay in the lab 
# 5 test

# Gripper (lynxmotion_ses_pro_model.link_visuals): the ellipsoid is centred 
# halfway along and sized so the finger-tip corners are inside
# (0.044/0.065)^2 + (0.0075/0.03)^2 + (0.055/0.08)^2 = 0.99 < 1.
GRIPPER_CENTRE = -(LEN_L1 / 2 + 0.055)  # along flange z (down is -)
GRIPPER_ELLIPSOID = (SE3(0, 0, GRIPPER_CENTRE), (0.065, 0.03, 0.08))
# Cookie centred on the TCP. Sized so the cookie's rim edge is inside:
# (0.03/0.04)^2 + (0.005/0.02)^2 = 0.63 < 1.
COOKIE_ELLIPSOID = (TOOL, (0.04, 0.04, 0.02))


def _cookie_cross():
    """Two diameters across the cookie at the TCP, as a polyline after the TCP."""
    return [TOOL * SE3(x, y, 0) for x, y in
            ((COOKIE_RADIUS, 0), (-COOKIE_RADIUS, 0), (0, COOKIE_RADIUS), (0, -COOKIE_RADIUS))]


LYNX_SETTINGS = PlannerSettings(
    tool=TOOL,
    roll=0.0, pitch=0.0,
    payload_up=(0, 0, 1), # the cookie's up is tool +z
    tcp_speed=TCP_SPEED,
    manip_epsilon=MANIP_EPSILON,
    via_min_radius=VIA_MIN_RADIUS,
    tool_chain=[TOOL],
    payload_chain=_cookie_cross(),
    tool_ellipsoids=[GRIPPER_ELLIPSOID],
    payload_ellipsoids=[COOKIE_ELLIPSOID],
)


class LynxPlanner(MotionPlanner):
    """The shared planner set up with the Lynxmotion's settings."""

    def __init__(self, robot, obstacles, min_link_z):
        super().__init__(robot, obstacles, min_link_z, LYNX_SETTINGS)
