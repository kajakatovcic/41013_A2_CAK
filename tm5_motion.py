# Motion settings for the OMRON TM5-700 plate station
#
# It uses the shared planner in motion_planner.py. This file only holds its 
# specifics, i.e. suction cup tool, tool-down grasp, speed and collision shapes
# of the tool and the plate.
#
# Picks the top plate from a plate dispenser with a suction cup and places
# it on the tray at the first conveyor stop.
#
#   jtraj used When nothing is near the tool so the uncontrolled path doesnt
#   matter.
#   RMRC used when the tool path is important, e.g. lowering to plate or
#   lifting/placing plate.

from math import pi

import numpy as np
from spatialmath import SE3

from motion_planner import MotionPlanner, PlannerSettings

# Suction tool- a 0.06x0.06x0.03m adapter block (vacuum generator) on
# the flange then the ir_support_extra_parts 'SuctionCup' scaled 1.8x
# The TCP is the centre of the cup's lip
ADAPTER_HEIGHT = 0.03
CUP_SCALE = 1.8
SUCTION_LENGTH = ADAPTER_HEIGHT + 0.057 * CUP_SCALE # 0.133 m
TOOL = SE3(0, 0, SUCTION_LENGTH)

# Tool pointing straight down: roll = pi, pitch = 0, yaw = bearing
# R = Rz(yaw) * Rx(pi): tool z points down and tool x points along the bearing
DOWN_ROLL = pi

# ir_support_extra_parts 'Plate' scaled 0.5
# The suction cup makes contact at PLATE_CONTACT
PLATE_RADIUS = 0.115
PLATE_HEIGHT = 0.0225
PLATE_CONTACT = 0.005
# Plate mid-height measured along tool z (down) from the TCP. 
# It is slightly above the TCP because the plate rim is above its centre
PLATE_MID = PLATE_CONTACT - PLATE_HEIGHT / 2

# RMRC tool speed cap. 
# Stays under the 0.25 m/s reduced-speed figure for collaborative operation
TCP_SPEED = 0.20 # m/s

# Manipulability threshold for DLS
# The TM5-700 is larger than the JAKA so its values are larger
MANIP_EPSILON = 0.01

# Collision shapes
# ELLIPSOIDS for the tool and plate: the plate is 0.23m wide and is lifted
# between guide posts and is very close. THe centre line says nothing about
# whether its rim hits a post. An ellipsoid would have the width (see lab 6)
# Line plane intersection also used as the obstacle side of the ellipsoid
# test is a point cloud which can be passed between points. 
# A cross of centre-lines across the plate (two diameters) is used

# Suction tool ellipsoid iscentred halfway along the tool to cover the
# adapter and cup
SUCTION_ELLIPSOID = (SE3(0, 0, SUCTION_LENGTH / 2), (0.04, 0.04, 0.075))
# Plate ellipsoid is flat + centred at the plate's mid-height. It is sized so the rim
# edge is inside:
# (0.115/0.13)^2 + (0.01125/0.03)^2 = 0.92 < 1.
PLATE_ELLIPSOID = (TOOL * SE3(0, 0, PLATE_MID), (0.13, 0.13, 0.03))


def _plate_cross():
    """Two diameters across the plate at mid-height, as a polyline after the TCP."""
    return [TOOL * SE3(x, y, PLATE_MID) for x, y in
            ((PLATE_RADIUS, 0), (-PLATE_RADIUS, 0), (0, PLATE_RADIUS), (0, -PLATE_RADIUS))]


TM5_SETTINGS = PlannerSettings(
    tool=TOOL,
    roll=DOWN_ROLL, pitch=0.0,
    payload_up=(0, 0, -1),# the plate's up is tool -z
    tcp_speed=TCP_SPEED,
    manip_epsilon=MANIP_EPSILON,
    via_min_radius=0.30,
    tool_chain=[TOOL],
    payload_chain=_plate_cross(),
    tool_ellipsoids=[SUCTION_ELLIPSOID],
    payload_ellipsoids=[PLATE_ELLIPSOID],
)


def suction_pose(base, r, phi, h):
    """
    World TCP pose with the suction cup pointing straight down at polar coords
    (r, phi, h) in the TM5 base frame. Same as MotionPlanner.pose_at with the
    TM5 settings. a plain function so props can be placed before a planner
    exists.
    """
    return base * SE3(r * np.cos(phi), r * np.sin(phi), h) * SE3.Rz(phi) * SE3.Rx(DOWN_ROLL)


class TM5Planner(MotionPlanner):
    """The shared planner set up with the TM5-700's settings"""

    def __init__(self, robot, obstacles, min_link_z):
        super().__init__(robot, obstacles, min_link_z, TM5_SETTINGS)
