# Motion settings for the JAKA MiniCobo coffee station.
#
# The planning methods themselves (jtraj, RMRC, collision checks, avoidance)
# are shared by every robot and live in motion_planner.py. This file holds only
# what is specific to the JAKA: its gripper, its side grasp, its speeds and
# the collision shapes of the gripper and the cup.
#
# The JAKA uses TWO trajectory methods based on the task:
#   jtraj (quintic, joint space) via plan_jtraj() for free space transits
#
#   RMRC (resolved motion rate control, Cartesian) via plan_rmrc() where tool path 
#   and orientation is critical
#
#   Measured on this robot (python coffee_demo.py --plan prints the table):
#       * jtraj between the same two poses sags 6-12mm BELOW the straight line.
#         The cup slides into the bay 5mm above the drip grate so jtraj would
#         drive it into the grate. RMRC stays within 0.1mm of the line.
#       * jtraj caps JOINT speed and not tool speed. On the full-cup carry its
#         peak TCP speed reached 0.28m/s (above the 0.25 m/s reduced-speed
#         figure). RMRC sets the TCP speed directly.
#       * Both kept the cup within 0.1deg of upright on these moves because
#         joints 2, 3 and 5 pitch in one plane. RMRC enforces it every step
#         rather than relying on that which matters if replanning is necessary

from math import pi

import numpy as np
from spatialmath import SE3

from motion_planner import MotionPlanner, PlannerSettings, DT 

# Gripper and grasp geometry
# A parallel two finger gripper with TCP 0.11m out along the flange z axis
# at the centre of the closed fingers
GRIPPER_LENGTH = 0.11
TOOL = SE3(0, 0, GRIPPER_LENGTH)

# Side grasped cup geometry. The cup is held from the side with the tool z-axis
# horizontal (pointing at cup) and tool x-axis pointing down. As long as tool x
# stays vertical the cup will stay upright regardless of yaw.
# MiniCobo joint 5's +-120 deg limit caps the tool-down envelope at ~0.30m above
# the base, which is why the JAKA does not use a tool-down grasp.
SIDE_PITCH = pi / 2

# Cup held by the gripper (12 oz takeaway cup, see coffee_station.py)
CUP_HEIGHT = 0.115
GRASP_HEIGHT = 0.055 # grasp point above the cup's base

# RMRC: tool-centre-point (TCP) speed cap. 0.15m/s carrying hot coffee is
# below the generally used 0.25m/s reduced-speed figure for cobot operation.
TCP_SPEED = 0.15 # m/s

# Manipulability threshold for switching to DLS.
# Lab 9 uses 0.1 for puma560. The MiniCobo is smaller so the threshold is scaled
# down (typical values along the task are 0.007-0.012)
MANIP_EPSILON = 0.004

# Collision ellipsoids (Lab 6 Q2) for the parts a centre-line cannot represent.
# Radii are along the TOOL axes: x points down (cup axis), y is the finger
# closing direction, z points forward from the flange.
#
# Why ellipsoids here: the gripper is 0.10 m wide and the cup is 0.09m wide.
# Their centrelines can pass 4-5cm from a wall that the real part would hit
# sp a line test would either miss those contacts or needs padding. An ellispid
# carries width.
# Why not ellipsoids alone: the obstacle side is a point cloud and a sampled
# surface can be passed between points. The centre-lines stay in the lab 5 test
# for this reason
#
# Gripper (coffee_station.Gripper): palm 0.07x0.10x0.04m, fingers reach
# 0.125m from the flange, closed fingers 0.05m either side of the axis. The
# ellipsoid is centred halfway along (0.0625m) and sized so the closed
# fingertip corners are inside:
# (0.015/0.05)^2 + (0.05/0.08)^2 + (0.0625/0.09)^2 = 0.96 < 1.
# It is still narrower than the 0.20m bay (0.08 < 0.10 each side).
GRIPPER_ELLIPSOID_OFFSET = 0.0625 # along tool z from the flange (m)
GRIPPER_ELLIPSOID_RADII = (0.05, 0.08, 0.09) # tool x, y, z (m)
# Cup (0.115m tall, rim radius 0.046m) centred at its mid-height.
# Sized so the rim edge is inside:
# (0.0575/0.085)^2 + (0.046/0.065)^2 = 0.96 < 1. The ellipsoid hangs a little
# below the cup's base. the surfaces the cup rests on are not
# obstacles so that does not read as a collision.
CUP_ELLIPSOID_RADII = (0.085, 0.065, 0.065)    # tool x (vertical), y, z (m)

JAKA_SETTINGS = PlannerSettings(
    tool=TOOL,
    roll=0.0, pitch=SIDE_PITCH,
    payload_up=(-1, 0, 0), # the cup's axis points along tool -x
    tcp_speed=TCP_SPEED,
    manip_epsilon=MANIP_EPSILON,
    via_min_radius=0.45, # side-grasp envelope starts ~0.45m out
    # lab 5 centre-lines: flange -> TCP then down the cup's axis from 1cm
    # above its base to its rim (tool +x points down the cup).
    tool_chain=[TOOL],
    payload_chain=[TOOL * SE3(GRASP_HEIGHT - 0.01, 0, 0),
                   TOOL * SE3(-(CUP_HEIGHT - GRASP_HEIGHT), 0, 0)],
    # Lab 6 ellipsoids: gripper always. cup centred at its mid-height which
    # is (GRASP_HEIGHT - CUP_HEIGHT/2) along tool x from the TCP.
    tool_ellipsoids=[(SE3(0, 0, GRIPPER_ELLIPSOID_OFFSET), GRIPPER_ELLIPSOID_RADII)],
    payload_ellipsoids=[(TOOL * SE3(GRASP_HEIGHT - CUP_HEIGHT / 2, 0, 0), CUP_ELLIPSOID_RADII)],
)


def side_grasp_pose(base, r, phi, h):
    """
    World TCP pose for a side grasp at polar coords (r, phi, h) in the robot
    BASE frame.
    r = horizontal distance from base axis
    phi = bearing from base x axis
    h = height above base plate
    Tool points radially outward so the cup is always approached head on.
    Same as MotionPlanner.pose_at with the JAKA's settings. kept as a plain
    function so the station props can be placed before a planner exists.
    """
    return base * SE3(r * np.cos(phi), r * np.sin(phi), h) * SE3.Rz(phi) * SE3.Ry(SIDE_PITCH)


def cup_tilt(T_tcp):
    """Angle (rad) between the held cup's axis and world vertical.
    The cup axis is the tool's -x axis (see SIDE GRASP above)"""
    R = T_tcp.A[:3, :3] if hasattr(T_tcp, "A") else np.asarray(T_tcp)[:3, :3]
    return float(np.arccos(np.clip(-R[2, 0], -1.0, 1.0)))


class JakaPlanner(MotionPlanner):
    """The shared planner set up with the JAKA's settings."""

    def __init__(self, robot, obstacles, min_link_z):
        super().__init__(robot, obstacles, min_link_z, JAKA_SETTINGS)
