# Shared Smooth Oper-Caterer workcell: 
# Includes enclosure, conveyor, tray, collection zone, safety equipment and areas
# for the other three robots.
#
# Explanation for each piece of equipment: RISK REGISTOR -> CONTROL -> MODEL PLACED
#
#  1. Attendee walks into a robot's swept volume.
#       Perimeter guarding: SafetyRailing on the long sides and barriers on the
#       end walls, every robot base at least reach + tool + payload inside it
#       (JAKA 0.74 m (@TODO include other robots here), see workcell_layout).
#  2. Attendee reaches into the cell through the conveyor exit (the one opening
#     that must stay open for trays to leave)
#       SafetyLightCurtain pair across the exit, on stands at belt height. A
#       break is an asynchronous sensor signal -> protective stop of robots AND
#       conveyor (safety_controller.LightCurtain). The tray is not stopped, it is
#       allowed to continue to the collection zone, where the attendee can take it.
#  3. Anyone sees a fault/hazard and needs to use the E-STOP.
#       Physical E-STOPs: a pedestal e-stop at the collection counter (where
#       attendees stand), a wall e-stop on the attendee railing next to the
#       operator, and one next to the staff gate.
#  4. Staff must enter to replenish any drinks/foods/cups or to service the cell.
#       SafetyGate on the staff side behind the coffee machine, with a
#       MagneticSwitch interlock. The cup dispenser faces this gate so refills
#       never need a reach past the arm.
#  5. People cannot tell whether the cell is running, stopped or faulted.
#       LightTower stack light at the exit corner, visible from the collection
#       zone and the viewing side and a beacon driven by the controller state.
#  6. Hot liquid and a heated appliance (coffee machine, etc.).
#       Hot-surface WarningSign at the staff gate, FireExtinguisher outside the
#       gate. Full cups move only under RMRC at a controlled speed limit and are
#       kept upright.
#  7. Trays and conveyor nip points at the collection counter.
#       CautionSign at the collection zone. The tray stops fully before the
#       attendee takes it and the light curtain separates the collection side.
#  8. Objects left in a robot's path (a milk pitcher on the bench).
#       Collision prediction and avoidance. 

from math import pi, ceil

import numpy as np
from spatialmath import SE3
from spatialgeometry import Cuboid, Cylinder, Sphere, Mesh
from ir_support_extra_parts import part_mesh, part_path

from workcell_layout import (
    Z_COUNTER, Z_BELT, CONVEYOR_Y, CONVEYOR_X_START, CONVEYOR_X_END, BELT_WIDTH,
    CELL_X_MIN, CELL_X_MAX, CELL_Y_MIN, CELL_Y_MAX, EXIT_X, EXIT_HALF_WIDTH,
    COLLECTION_X, COLLECTION_ZONE_SIZE, STATIONS, LIGHT_CURTAIN_Z,
    ESTOP_COLLECTION_POSE, ESTOP_CONSOLE_POSE, STAFF_GATE_X, TRAY_START_X,
)

HIDDEN = SE3(0, 0, -5) # parking pose under the floor


class Indicator:
    """
    A lamp or field that changes colour at run time.
    The Swift build in this course cannot recolour a shape once it has been
    added: setting .color sends a 'shape_update' message that the browser page
    has no handler for, and env.step() then times out. Pose updates DO work, so
    an indicator is one shape per colour, and switching colour moves the chosen
    shape into place and parks the others under the floor.
    """

    def __init__(self, make_shape, colours, pose, initial):
        self.pose = pose
        self.shapes = {name: make_shape(rgba) for name, rgba in colours.items()}
        self.set(initial)

    def set(self, name):
        self.current = name
        for key, shape in self.shapes.items():
            shape.T = (self.pose if key == name else HIDDEN).A

    def add_to_env(self, env):
        for shape in self.shapes.values():
            env.add(shape)


LAMP_COLOURS = {
    "green": (0.1, 0.85, 0.2, 0.95),
    "amber": (1.0, 0.65, 0.0, 0.95),
    "red": (0.95, 0.05, 0.05, 0.95),
    "blue": (0.2, 0.5, 1.0, 0.95),
    "grey": (0.4, 0.4, 0.4, 1.0),
}


def _run(part, part_len, a, b, fixed, along_x, extra=None):
    """
    Place copies of a straight guarding part to cover [a, b] on one line.
    """
    n = max(1, ceil((b - a) / part_len - 1e-6))
    centres = [(a + b) / 2] if n == 1 else np.linspace(a + part_len / 2, b - part_len / 2, n)
    out = []
    for c in centres:
        pose = SE3(c, fixed, 0) if along_x else SE3(fixed, c, 0) * SE3.Rz(pi / 2)
        out.append(part_mesh(part, pose=pose * (extra or SE3())))
    return out

# Tray and conveyor

class Tray:
    """
    Serving tray on the conveyor - ir_support_extra_parts 'Tray' scaled to
    0.475 x 0.35 m. Frame at the centre of its base. Items placed on it are
    attached at a fixed local offset and move with it.
    """
    SCALE = [1.25, 1.4, 1.0]
    FLOOR = 0.006
    # Fixed slots, in the tray frame.
    PLATE_SLOT = SE3(-0.11, 0.0, FLOOR)
    DRINK_SLOT = SE3(0.13, 0.10, FLOOR)
    CUP_SLOT = SE3(0.12, -0.04, FLOOR)

    def __init__(self, x):
        self.mesh = Mesh(str(part_path("Tray")), scale=self.SCALE, color=(0.20, 0.35, 0.55, 1.0))
        self.attached = [] # (object with set_pose, local SE3)
        # Placeholders for what the upstream robots deliver (TM12 plate,
        # Lynxmotion food, FR3 drink) until their tasks are integrated. @TODO
        plate = Mesh(str(part_path("Plate")), scale=[0.5, 0.5, 0.5])
        food = part_mesh("Lunchbox")
        drink = part_mesh("JuiceBoxOrange")
        self._meshes = [self.mesh, plate, food, drink]
        self.attach(_MeshItem(plate), self.PLATE_SLOT)
        self.attach(_MeshItem(food), self.PLATE_SLOT * SE3(0, 0, 0.012))
        self.attach(_MeshItem(drink), self.DRINK_SLOT)
        self.set_x(x)

    def attach(self, item, local):
        self.attached.append((item, local))
        if hasattr(self, "pose"):
            item.set_pose(self.pose * local)

    def set_x(self, x):
        self.x = x
        self.pose = SE3(x, CONVEYOR_Y, Z_BELT)
        self.mesh.T = self.pose.A
        for item, local in self.attached:
            item.set_pose(self.pose * local)

    def slot_world(self, slot, x=None):
        """
        World pose of a tray slot with the tray at x (default: where it is).
        """
        return SE3(self.x if x is None else x, CONVEYOR_Y, Z_BELT) * slot

    def add_to_env(self, env):
        for m in self._meshes:
            env.add(m)


class _MeshItem:
    """Allows a bare Mesh to be attached to the tray. It has a set_pose() method that moves the mesh."""
    def __init__(self, mesh):
        self.mesh = mesh

    def set_pose(self, pose):
        self.mesh.T = pose.A


class Conveyor:
    """Belt conveyor built from primitives (cuboids)."""

    def __init__(self):
        self.shapes = []
        length = CONVEYOR_X_END - CONVEYOR_X_START
        xc = (CONVEYOR_X_END + CONVEYOR_X_START) / 2
        belt_t = 0.03
        self.shapes.append(Cuboid(scale=[length, BELT_WIDTH, belt_t], color=(0.10, 0.10, 0.11, 1.0),
                                  pose=SE3(xc, CONVEYOR_Y, Z_BELT - belt_t / 2)))
        # Side frames with a 15mm lip above the belt to locate the tray.
        # These are kept low on purpose as the JAKA's wrist passes over the near frame when
        # it lowers a cup onto the tray.
        for side in (-1, 1):
            y = CONVEYOR_Y + side * (BELT_WIDTH / 2 + 0.02)
            self.shapes.append(Cuboid(scale=[length, 0.04, 0.115], color=(0.62, 0.64, 0.67, 1.0),
                                      pose=SE3(xc, y, Z_BELT + 0.015 - 0.115 / 2)))
            for x in np.arange(CONVEYOR_X_START + 0.1, CONVEYOR_X_END, 1.6):
                self.shapes.append(Cuboid(scale=[0.05, 0.05, Z_BELT - 0.10], color=(0.45, 0.46, 0.48, 1.0),
                                          pose=SE3(x, y, (Z_BELT - 0.10) / 2)))
        for x in (CONVEYOR_X_START, CONVEYOR_X_END):
            self.shapes.append(Cylinder(radius=0.035, length=BELT_WIDTH, color=(0.5, 0.5, 0.5, 1.0),
                                        pose=SE3(x, CONVEYOR_Y, Z_BELT - 0.035) * SE3.Rx(pi / 2)))
        self.tray = Tray(TRAY_START_X)

    def add_to_env(self, env):
        for s in self.shapes:
            env.add(s)
        self.tray.add_to_env(env)


# The whole shared cell @TODO: add the other robots and their props when they are integrated
class Workcell:
    """
    Everything except the robots and their station props. Holds handles the
    controller needs at run time: the conveyor/tray, the light curtain field and
    intruder, and the stack-light beacon.
    """

    def __init__(self):
        self.shapes = []
        self.conveyor = Conveyor()
        self._floor()
        self._enclosure()
        self._light_curtain()
        self._estops_and_indicators()
        self._signage_and_people()
        self._reserved_stations()

    # floor
    def _floor(self):
        self.shapes.append(Cuboid(scale=[CELL_X_MAX - CELL_X_MIN, CELL_Y_MAX - CELL_Y_MIN, 0.004],
                                  color=(0.80, 0.80, 0.78, 1.0),
                                  pose=SE3((CELL_X_MAX + CELL_X_MIN) / 2, (CELL_Y_MAX + CELL_Y_MIN) / 2, 0.002)))
        # Collection zone marking outside the guarding
        zx, zy = COLLECTION_ZONE_SIZE
        self.shapes.append(Cuboid(scale=[zx, zy, 0.006], color=(0.98, 0.80, 0.10, 1.0),
                                  pose=SE3(COLLECTION_X, CONVEYOR_Y, 0.003)))

    # Perimeter guarding and staff gate 
    def _enclosure(self):
        rail_len, barrier_len = 2.565, 1.52
        # Attendee side: full length guarding
        self.shapes += _run("SafetyRailing", rail_len, CELL_X_MIN, CELL_X_MAX, CELL_Y_MAX, True)
        # Staff side: up to the gate
        gate_half = 1.055 / 2
        self.shapes += _run("SafetyRailing", rail_len, CELL_X_MIN, STAFF_GATE_X - gate_half, CELL_Y_MIN, True)
        self.shapes.append(part_mesh("SafetyGate", pose=SE3(STAFF_GATE_X, CELL_Y_MIN, 0)))
        # Interlock switch on the gate's latch post
        self.shapes.append(part_mesh("MagneticSwitch", pose=SE3(STAFF_GATE_X + gate_half - 0.03, CELL_Y_MIN + 0.12, 1.0)))
        # Upstream end wall
        self.shapes += _run("barrier1.5x0.2x1m", barrier_len, CELL_Y_MIN, CELL_Y_MAX, CELL_X_MIN, False)
        # Exit end wall with the conveyor opening between the curtain posts
        self.shapes += _run("barrier1.5x0.2x1m", barrier_len, CELL_Y_MIN, -EXIT_HALF_WIDTH, EXIT_X, False)
        self.shapes += _run("barrier1.5x0.2x1m", barrier_len, EXIT_HALF_WIDTH, CELL_Y_MAX, EXIT_X, False)

    # Light curtain
    def _light_curtain(self):
        z0, z1 = LIGHT_CURTAIN_Z
        for side in (-1, 1):
            y = CONVEYOR_Y + side * EXIT_HALF_WIDTH
            self.shapes.append(Cuboid(scale=[0.08, 0.08, z0], color=(0.2, 0.2, 0.2, 1.0),
                                      pose=SE3(EXIT_X, y, z0 / 2)))
            self.shapes.append(part_mesh("SafetyLightCurtain", pose=SE3(EXIT_X, y, z0)))
        # The detection field between the posts coloured by the sensor state.
        field_scale = [0.01, 2 * EXIT_HALF_WIDTH - 0.12, z1 - z0]
        self.curtain_field = Indicator(
            lambda rgba: Cuboid(scale=field_scale, color=rgba),
            {"green": (0.1, 0.9, 0.2, 0.15), "red": (0.95, 0.1, 0.1, 0.35)},
            SE3(EXIT_X, CONVEYOR_Y, (z0 + z1) / 2), "green")
        # Simulate an intruder via an attendee's hand reaching into the exit
        self.hand_in_pose = SE3(EXIT_X + 0.10, CONVEYOR_Y + 0.05, Z_BELT + 0.25) * SE3.Rz(pi)
        self.hand = part_mesh("hand", pose=HIDDEN)
        self.shapes.append(self.hand)
        self.hand_inside = False

    def toggle_hand(self):
        self.hand_inside = not self.hand_inside
        self.hand.T = (self.hand_in_pose if self.hand_inside else HIDDEN).A
        return self.hand_inside

    def hand_box(self):
        """World AABB (lo, hi) of the hand mesh: x -0.09..0.14, y -0.12..0.16,
        z 0.03..0.11 in its own frame."""
        T = SE3(self.hand.T, check=False)
        corners = np.array([(T * SE3(x, y, z)).t for x in (-0.092, 0.143)
                            for y in (-0.115, 0.16) for z in (0.033, 0.106)])
        return corners.min(axis=0), corners.max(axis=0)

    # e-stops, and stack light ---------------------------------------------
    def _estops_and_indicators(self):
        # Pedestal e-stop at the collection counter on a stand at counter height
        p = ESTOP_COLLECTION_POSE
        self.shapes.append(Cuboid(scale=[0.25, 0.25, Z_COUNTER - 0.29], color=(0.85, 0.75, 0.1, 1.0),
                                  pose=SE3(p.t[0], p.t[1], (Z_COUNTER - 0.29) / 2)))
        self.shapes.append(part_mesh("emergencyStopButton", pose=SE3(p.t[0], p.t[1], Z_COUNTER - 0.29)))
        # Wall e-stops at attendee railing (operator), and beside the staff gate
        self.shapes.append(part_mesh("emergencyStopWallMounted", pose=ESTOP_CONSOLE_POSE))
        self.shapes.append(part_mesh("emergencyStopWallMounted",
                                     pose=SE3(STAFF_GATE_X - 0.75, CELL_Y_MIN, 1.0) * SE3.Rz(pi) * SE3.Rx(pi / 2)))
        # Stack light on a pole at the exit corner seen from the collection zone
        pole_xy = (EXIT_X - 0.15, EXIT_HALF_WIDTH + 0.35)
        pole_h = 1.5
        self.shapes.append(Cuboid(scale=[0.05, 0.05, pole_h], color=(0.3, 0.3, 0.3, 1.0),
                                  pose=SE3(pole_xy[0], pole_xy[1], pole_h / 2)))
        self.shapes.append(part_mesh("LightTower", pose=SE3(pole_xy[0], pole_xy[1], pole_h)))
        self.beacon = Indicator(lambda rgba: Sphere(radius=0.07, color=rgba),
                                {k: LAMP_COLOURS[k] for k in ("green", "amber", "red", "blue")},
                                SE3(pole_xy[0], pole_xy[1], pole_h + 0.573 + 0.07), "amber")

    # Signage, extinguisher, people
    def _signage_and_people(self):
        self.shapes.append(part_mesh("WarningSign", pose=SE3(STAFF_GATE_X - 1.0, CELL_Y_MIN - 0.45, 0) * SE3.Rz(-pi / 2)))
        self.shapes.append(part_mesh("FireExtinguisher", pose=SE3(STAFF_GATE_X + 0.75, CELL_Y_MIN - 0.35, 0)))
        self.shapes.append(part_mesh("CautionSign", pose=SE3(COLLECTION_X + 0.55, -0.95, 0) * SE3.Rz(pi)))
        # Attendee collecting, an onlooker, and a staff member outside the gate
        self.shapes.append(part_mesh("personFemaleBusiness", pose=SE3(COLLECTION_X + 0.1, 1.05, 0) * SE3.Rz(-pi / 2)))
        self.shapes.append(part_mesh("personMaleCasual", pose=SE3(-0.6, CELL_Y_MAX + 0.7, 0) * SE3.Rz(-pi / 2)))
        self.shapes.append(part_mesh("SafetyPerson", pose=SE3(STAFF_GATE_X, CELL_Y_MIN - 0.8, 0) * SE3.Rz(pi / 2)))

    # Reserved stations for other robots (TM12, Lynxmotion, FR3) @TODO: add props when they are integrated
    def _reserved_stations(self):
        """Translucent floor pads marking where teammates' robots will mount."""
        for name, (pose, _task) in STATIONS.items():
            if name == "JAKA":
                continue
            self.shapes.append(Cuboid(scale=[0.9, 0.9, 0.005], color=(0.3, 0.5, 0.9, 0.35),
                                      pose=SE3(pose.t[0], pose.t[1], 0.006)))

    def add_to_env(self, env):
        for s in self.shapes:
            env.add(s)
        self.curtain_field.add_to_env(env)
        self.beacon.add_to_env(env)
        self.conveyor.add_to_env(env)
