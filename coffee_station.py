# JAKA MiniCobo coffee station: bench, cup dispenser, coffee machine, cup,
# gripper visual and the movable "deliberate obstacle" (a milk pitcher).
#
# Frames: Every prop is placed from the robot's BASE frame using the same polar
# convention as the motion planner, side_grasp_pose(base, r, phi, h):
#     r   = distance from the base axis, phi = bearing from the base x-axis,
#     h   = height above the base plate (= the counter top).
# So the cup sits exactly where the planner will reach for it, by construction.
# Move or rotate the JAKA's base in workcell_layout.STATIONS and the whole
# station follows.
#
#                      +x (conveyor direction)
#           dispenser            machine bay
#           phi = -90 deg  JAKA  phi = 0 deg
#                    \      |      /
#                       tray slot on the conveyor, phi = +90 deg
#
# The JAKA's side-grasp envelope is a ring about 0.45-0.65m from its base, at
# 0.05-0.30m above it (measured by IK sweep) so all three stations sit on that
# ring at counter height. The dispenser is on the staff side (-y) so staff refill
# cups through the access gate without reaching past the arm, and every swing
# between stations passes behind the robot, never over the attendee side.

from math import pi

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Sphere, Mesh
from ir_support_extra_parts import part_mesh, part_path

from collisions import Obstacle
from jaka_motion import side_grasp_pose
from workcell_scene import Indicator, LAMP_COLOURS, HIDDEN


# Station geometry, in the JAKA base frame (r, phi, h). TCP heights are the cup
# GRASP point: surface height + GRASP_HEIGHT.
# TakeawayCup mesh is 0.204 x 0.256m
# Scaled to a 12oz cup (0.115m tall, 0.092m rim)
CUP_SCALE = 0.45 
CUP_HEIGHT = 0.256 * CUP_SCALE  
CUP_RIM_RADIUS = 0.102 * CUP_SCALE
CUP_GRIP_RADIUS = 0.082 * CUP_SCALE  # cup radius at the grasp height
GRASP_HEIGHT = 0.055 # grasp point above the cup's base

PAD_TOP = 0.010 # dispenser pad surface above the counter
DRIP_TOP = 0.020 # coffee machine drip tray surface
TRAY_FLOOR = 0.010 # tray floor above the belt (= workcell_scene.Tray.FLOOR)

PHI_DISPENSER = -pi / 2
PHI_MACHINE = 0.0
PHI_TRAY = pi / 2

R_PICK = 0.56 # cup centre on the dispenser pad
R_BAY = 0.60 # cup centre under the coffee spout
R_TRAY = 0.58 # cup slot on the tray
R_CLEAR = 0.45 # standoff for straight-line approach/withdraw
R_READY = 0.45

H_PICK = PAD_TOP + GRASP_HEIGHT
H_LIFT = H_PICK + 0.05
BAY_HOVER = 0.005 # the cup is held 5mm above the drip grate while it
                  # slides in, pours coffee and slides out (never set down)
H_BAY = DRIP_TOP + BAY_HOVER + GRASP_HEIGHT
H_TRAY = TRAY_FLOOR + GRASP_HEIGHT
# carry height - cup base 0.085m above the counter 
# and clear of the tray rim and the drink on the tray
H_CARRY = 0.14 
H_READY = 0.30

# Named TCP targets as (r, phi, h), used by the task and by the props below
POSES = {
    "ready":      (R_READY, -pi / 4, H_READY),
    "pre_pick":   (R_CLEAR, PHI_DISPENSER, H_PICK),
    "pick":       (R_PICK, PHI_DISPENSER, H_PICK),
    "lift":       (R_PICK, PHI_DISPENSER, H_LIFT),
    "pre_insert": (R_CLEAR + 0.01, PHI_MACHINE, H_BAY),
    "insert":     (R_BAY, PHI_MACHINE, H_BAY),
    "above_tray": (R_TRAY, PHI_TRAY, H_CARRY),
    "on_tray":    (R_TRAY, PHI_TRAY, H_TRAY),
    "retreat":    (R_CLEAR, PHI_TRAY, H_TRAY),
}

# Seeds for ikine_LM. q2 + q3 + q5 = -pi/2 keeps the tool horizontal (side
# grasp). joint 1 is replaced with the target bearing when used.
IK_SEEDS = [
    np.array([0, -0.5, -1.0, 0, -0.07, 0]),
    np.array([0, -1.2, -1.4, 0, 1.0, 0]),
    np.array([0, -1.0, -2.0, 0, 1.4, 0]),
]

# Obstacles are the props' real sizes. The width of the gripper and the cup is 
# handled by the planner's tool ellipsoids (see jaka_motion.py) so the obstacles
# do not need to be grown to cover them.

def polar_point(base, r, phi, h):
    """World position of base-frame polar coordinates."""
    return (base * SE3(r * np.cos(phi), r * np.sin(phi), h)).t


def make_obstacle(name, T, centre, size):
    """
    Collision box for a prop part. It is the same size as its visual Cuboid at the
    same pose (prop frame T times the part's local centre). The box is turned
    with the prop so that the collision test is exact.
    """
    return Obstacle(name, size, T * SE3(*centre))


# Cup and gripper
class Cup:
    """
    ir_support_extra_parts 'TakeawayCup' scaled by 0.45 to a 12oz cup.
    Frame at the centre of its base. While held its pose is derived from the
    TCP - the cup hangs GRASP_HEIGHT below the TCP along the tool x-axis (which
    points down) rotated so the cup's z is world-up.
    """
    IN_TCP = SE3(GRASP_HEIGHT, 0, 0) * SE3.Ry(-pi / 2)

    def __init__(self, pose=None):
        self.mesh = Mesh(str(part_path("TakeawayCup")), scale=[CUP_SCALE] * 3,
                         color=(0.93, 0.90, 0.84, 1.0))
        self.set_pose(pose if pose is not None else SE3(0, 0, -5))

    def set_pose(self, pose):
        self.pose = pose
        self.mesh.T = pose.A

    def held_by(self, T_tcp):
        self.set_pose(T_tcp * self.IN_TCP)

    def add_to_env(self, env):
        env.add(self.mesh)


class Gripper:
    """
    Visual-only parallel gripper on the JAKA flange (palm + two fingers).
    Tool frame: z forward, x down (side grasp), fingers close along y.
    """
    PALM = (0.07, 0.10, 0.04)
    FINGER = (0.03, 0.012, 0.085)
    OPEN = 0.065 # finger inner face from the TCP axis
    CLOSED = CUP_GRIP_RADIUS + 0.001

    def __init__(self):
        self.palm = Cuboid(scale=list(self.PALM), color=(0.25, 0.27, 0.30, 1.0))
        self.fingers = [Cuboid(scale=list(self.FINGER), color=(0.70, 0.72, 0.75, 1.0))
                        for _ in range(2)]
        self.opening = self.OPEN

    def update(self, T_flange):
        self.palm.T = (T_flange * SE3(0, 0, self.PALM[2] / 2)).A
        z = self.PALM[2] + self.FINGER[2] / 2
        y = self.opening + self.FINGER[1] / 2
        self.fingers[0].T = (T_flange * SE3(0, y, z)).A
        self.fingers[1].T = (T_flange * SE3(0, -y, z)).A

    def add_to_env(self, env):
        env.add(self.palm)
        for f in self.fingers:
            env.add(f)


# Station fixtures
class CoffeeStation:
    """
    Builds the bench and the props around the JAKA and owns their collision
    obstacles. 
    'obstacles' is the list handed to the planner.
    """

    def __init__(self, base):
        self.base = base
        self.shapes = []
        self.obstacles = []

        self._build_bench()
        self._build_dispenser()
        self._build_machine()
        self._build_pitcher()

        self.gripper = Gripper()
        self.cup = Cup()

    # Bench  
    def _build_bench(self):
        """
        ir_support_extra_parts 'Workbench' scaled to 1.6x1.125m so the dispenser
        behind the robot is still on the bench. Its top (0.93m) is the counter
        height Z_COUNTER. The JAKA base plate sits on it.
        """
        bx, by = self.base.t[0], self.base.t[1]
        bench_centre = SE3(bx + 0.05, by - 0.1875, 0)
        self.bench = Mesh(str(part_path("Workbench")), scale=[0.64, 1.5, 1.0],
                          pose=bench_centre)
        self.shapes.append(self.bench)

    # Cup dispenser
    def _build_dispenser(self):
        """
        A cup dispenser - a pad where the next cup is dropped, a body behind it
        holding the cup stack, and spare cups on top. 
        Frame at the pad centre, x pointing away from the robot.
        """
        phi, r = PHI_DISPENSER, R_PICK
        self.T_dispenser = self.base * SE3.Rz(phi) * SE3(r, 0, 0)
        T = self.T_dispenser

        pad = Cuboid(scale=[0.12, 0.12, PAD_TOP], color=(0.15, 0.15, 0.15, 1.0),
                     pose=T * SE3(0, 0, PAD_TOP / 2))
        # The body's front face is 0.07m behind the cup's centre. The real cup
        # rim reaches 0.046m but its collision ellipsoid reaches 0.065m
        # (an ellipsoid has to be wider than the cylinder to contain its rim),
        # so the body sits 0.07m back to leave the ellipsoid clear.
        body_c, body_s = (0.12, 0, 0.25), (0.10, 0.16, 0.50)
        body = Cuboid(scale=list(body_s), color=(0.80, 0.82, 0.85, 1.0),
                      pose=T * SE3(*body_c))
        chute = Cuboid(scale=[0.005, 0.11, 0.30], color=(0.6, 0.8, 1.0, 0.35),
                       pose=T * SE3(0.0675, 0, 0.30))
        self.shapes += [pad, body, chute]
        # Spare cup stack on top of the dispenser 
        for i in range(4):
            stack_cup = Mesh(str(part_path("TakeawayCup")), scale=[CUP_SCALE] * 3,
                             color=(0.93, 0.90, 0.84, 1.0),
                             pose=T * SE3(0.12, 0, 0.50 + 0.018 * i))
            self.shapes.append(stack_cup)

        # Collision box covers the body and the spare cups stacked on top
        stack_top = body_s[2] + 0.018 * 3 + CUP_HEIGHT
        self.obstacles.append(make_obstacle("cup dispenser", T,
                                            (body_c[0], 0, stack_top / 2),
                                            (body_s[0], body_s[1], stack_top)))

    @property
    def cup_dispense_pose(self):
        return self.T_dispenser * SE3(0, 0, PAD_TOP)

    # Coffee machine
    def _build_machine(self):
        """
        Bean-to-cup machine built from primitives. Frame at the centre of its FRONT face 
        on the counter, x pointinginto the machine. The dispensing bay is a 0.20m wide, 
        0.16m deep, 0.18m tall recess. It has walls on both sides, the housing behind, and 
        the brew head above. The cup must go in and out on a straight horizontal line,
        hence why this move uses RMRC.
        """
        # The front face is 0.06m in front of the cup's brewing position. The
        # cup waits at R_CLEAR + 0.01 = 0.46m before going in and swings past
        # the bay's front corners on the way in and out. Its collision
        # ellipsoid reaches 0.065m ahead of the cup centre (0.525m), so the
        # front face at 0.54m leaves it 1.5cm clear of the side walls.
        r_front = R_BAY - 0.06
        self.T_machine = self.base * SE3.Rz(PHI_MACHINE) * SE3(r_front, 0, 0)
        T = self.T_machine
        steel, black = (0.72, 0.74, 0.77, 1.0), (0.12, 0.12, 0.13, 1.0)

        depth, half_w, height = 0.30, 0.16, 0.42
        bay_d, bay_hw, head_z = 0.16, 0.10, 0.20

        parts = {
            # name: (centre, size, colour, is_obstacle)
            "plinth": ((depth / 2, 0, DRIP_TOP / 2), (depth, 2 * half_w, DRIP_TOP), black, False),
            "drip grate": ((bay_d / 2, 0, DRIP_TOP + 0.001), (bay_d - 0.02, 2 * bay_hw - 0.02, 0.002), (0.35, 0.35, 0.36, 1.0), False),
            "machine housing": (((bay_d + depth) / 2, 0, (DRIP_TOP + height) / 2), (depth - bay_d, 2 * half_w, height - DRIP_TOP), steel, True),
            "machine left wall": ((bay_d / 2, (bay_hw + half_w) / 2, (DRIP_TOP + height) / 2), (bay_d, half_w - bay_hw, height - DRIP_TOP), steel, True),
            "machine right wall": ((bay_d / 2, -(bay_hw + half_w) / 2, (DRIP_TOP + height) / 2), (bay_d, half_w - bay_hw, height - DRIP_TOP), steel, True),
            "brew head": ((bay_d / 2, 0, (head_z + height) / 2), (bay_d, 2 * bay_hw, height - head_z), black, True),
        }
        for name, (c, s, col, is_obs) in parts.items():
            self.shapes.append(Cuboid(scale=list(s), color=col, pose=T * SE3(*c)))
            if is_obs:
                self.obstacles.append(make_obstacle(name, T, c, s))

        hopper = Cylinder(radius=0.06, length=0.12, color=(0.35, 0.22, 0.12, 0.6),
                          pose=T * SE3(0.23, 0, height + 0.06))
        spout = Cylinder(radius=0.012, length=0.03, color=(0.5, 0.5, 0.5, 1.0),
                         pose=T * SE3(0.08, 0, head_z - 0.015))
        self.shapes += [hopper, spout]

        # Status light on the brew head and the coffee stream (shown while brewing).
        self.machine_light = Indicator(lambda rgba: Sphere(radius=0.012, color=rgba),
                                       {k: LAMP_COLOURS[k] for k in ("grey", "amber", "green")},
                                       T * SE3(-0.001, 0.07, 0.36), "grey")
        rim_z = DRIP_TOP + BAY_HOVER + CUP_HEIGHT
        self.stream = Cylinder(radius=0.004, length=head_z - 0.03 - rim_z,
                               color=(0.30, 0.17, 0.08, 1.0), pose=SE3(0, 0, -5))
        self._stream_pose = T * SE3(0.08, 0, (head_z - 0.03 + rim_z) / 2)
        self.shapes.append(self.stream)

    def set_brewing(self, brewing, done=False):
        self.stream.T = (self._stream_pose if brewing else HIDDEN).A
        self.machine_light.set("amber" if brewing else ("green" if done else "grey"))

    # Deliberate obstacle for GUI interactivity
    def _build_pitcher(self):
        """
        A milk pitcher a staff member might leave on the bench. This is the controlled
        object that can deliberately placed in the planned carry path (GUI toggle). 
        Parked at the far end of the bench otherwise.
        """
        self.pitcher_parked = SE3(*polar_point(self.base, 0.55, 2.7, 0))
        # In the path- on the bench between the machine and the tray under the
        # middle of the full-cup carry arc.
        self.pitcher_in_path = SE3(*polar_point(self.base, 0.50, pi / 5, 0))
        self.pitcher_mesh = part_mesh("MilkPitcher", pose=self.pitcher_parked)
        self.pitcher = Obstacle("milk pitcher", [0.09, 0.125, 0.11],
                                self.pitcher_parked * SE3(0, 0, 0.055), movable=True)
        self.obstacles.append(self.pitcher)
        self.shapes.append(self.pitcher_mesh)
        self.pitcher_placed = False

    def toggle_pitcher(self):
        """Move the pitcher into/out of the carry path. Returns the new state."""
        self.pitcher_placed = not self.pitcher_placed
        pose = self.pitcher_in_path if self.pitcher_placed else self.pitcher_parked
        self.pitcher_mesh.T = pose.A
        self.pitcher.move_to(pose * SE3(0, 0, 0.055))
        return self.pitcher_placed

    def tcp_pose(self, name):
        """World TCP pose of a named station target."""
        return side_grasp_pose(self.base, *POSES[name])

    def add_to_env(self, env):
        for s in self.shapes:
            env.add(s)
        self.machine_light.add_to_env(env)
        self.gripper.add_to_env(env)
        self.cup.add_to_env(env)
