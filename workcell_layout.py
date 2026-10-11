# Shared world layout for the Smooth Oper-Caterer workcell.
#
# EVERY position in the cell is defined here in ONE world frame, so the four
# robots, the conveyor, the props and the safety sensors cannot drift out of
# agreement. Each robot mounts at STATIONS[...] and places props
# relative to that frame.
#
# World frame has origin on the floor at the middle of the conveyor's run inside the
# cell. +X is the direction the conveyor travels (towards the collection zone),
# +Z is up, +Y points towards the attendee viewing side.
#
#                         attendees / viewing side (+Y)
#     ---------------------------------------------------------------- railing
#     |                                                              |  light
#     | [TM5-700]    [Lynxmotion]     [FAIRINO FR3]      [JAKA]       |  curtain   collection
#  ==tray==>=======================conveyor==========================||=======>=  zone
#     |  plates       food            drinks            coffee        |
#     ---------------------------------------------------------------- railing
#                          staff access gate (-Y)
#
# The hot drink is added LAST (by the JAKA MiniCobo) so it spends
# the least time travelling, and the coffee station is closest to the exit.

from math import pi
from spatialmath import SE3

# Counter height. The serving bench, the conveyor belt and the collection
# counter all sit at this height so a tray never has to change level.
Z_COUNTER = 0.93
# Conveyor (runs along +X at y = 0)
CONVEYOR_Y = 0.0
CONVEYOR_X_START = -4.2 # upper end behind the TM5-700 plate station
CONVEYOR_X_END = 3.9 # downstream end inside the collection zone
BELT_WIDTH = 0.45 # carries a 0.35 m wide tray with a margin
Z_BELT = Z_COUNTER # top surface of the belt
BELT_SPEED = 0.20 # m/s (a walking-pace conveyor speed)

# Enclosure: Railings run the full length on both sides. The conveyor leaves
# the cell through an opening in the +X end wall which is guarded by the light curtain.
# Every robot base is at least its reach + tool + payload clear of the railing
# e.g. (JAKA: 0.58 m reach + 0.11 m gripper + 0.05 m cup = 0.74 m).
CELL_X_MIN = -4.6
CELL_X_MAX = 2.75 # exit wall (conveyor passes through here)
CELL_Y_MIN = -1.85 # staff side
CELL_Y_MAX = 1.85 # attendee side

EXIT_X = CELL_X_MAX # Conveyor exit opening in the +X wall. The light curtain spans this gap.
EXIT_HALF_WIDTH = 0.33 # curtain posts sit just outside the belt edges

# Collection zone outside the cell and past the light curtain. The tray halts here for the 
# attendee to collect the order.
COLLECTION_X = 3.40 # tray stops here for the attendee
COLLECTION_ZONE_SIZE = (1.30, 1.40) # floor marking

# Robot stations: base frame of each robot in the world frame.
# Robots sit on the -Y (staff) side of the belt, facing the belt.
# @TODO: add other robots (Lynxmotion, FR3) when they are integrated
JAKA_X = 1.60
JAKA_Y = -0.62 # 0.58 m from the tray's coffee slot

# OMRON TM5-700 (extra scene robot, ir_support_extra_robots). Bench-mounted at
# counter height like the JAKA: with a 0.7 m reach it cannot serve the tray
# from the floor. Base clearance to the railing: 0.7 m reach + 0.133 m suction
# tool + 0.115 m plate radius = 0.95 m (staff railing is 1.3 m away).
TM5_X = -3.40
TM5_Y = -0.55 # 0.55 m from the tray's plate slot

STATIONS = {
    # name:        (base pose in world,                       task)
    "TM5700":      (SE3(TM5_X, TM5_Y, Z_COUNTER),             "places a plate on the tray"),
    "Lynxmotion":  (SE3(-1.70, -0.75, Z_COUNTER),             "serves food onto the plate"),
    "FAIRINO_FR3": (SE3(-0.05, -0.75, Z_COUNTER),             "places a cold drink on the tray"),
    "JAKA":        (SE3(JAKA_X, JAKA_Y, Z_COUNTER),           "brews and places the coffee"),
}

# Where the tray halts on the belt for each station (x of the tray centre).
# JAKA: Halt position calculated so the coffee slot is aligned with the JAKA gripper at its home pose.
TRAY_START_X = -0.05 # tray enters the JAKA demo at the FR3 station, already
                     # holding the plate and drink from further up
TRAY_ENTRY_X = -3.95 # empty tray enters at the upstream end (TM5 demo, main scene)

# Safety equipment positions (world frame), chosen from the risk assessment in
# workcell_scene.py
LIGHT_CURTAIN_X = EXIT_X # vertical plane across the exit
LIGHT_CURTAIN_Z = (Z_BELT, Z_BELT + 0.816) # SafetyLightCurtain post height
ESTOP_COLLECTION_POSE = SE3(3.40, -0.95, 0.0) # pedestal e-stop
# wall-mounted e-stop on the attendee-side railing
ESTOP_CONSOLE_POSE = SE3(0.80, CELL_Y_MAX, 1.00) * SE3.Rx(pi / 2)
STAFF_GATE_X = CELL_X_MAX - 0.53 # interlocked gate behind the coffee
                                 # machine, for refilling cups and beans
