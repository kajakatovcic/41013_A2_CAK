import os
import trimesh
import numpy as np
import shutil, tempfile
import roboticstoolbox as rtb
from roboticstoolbox import Link, ET, ERobot
from spatialmath import SE3
from spatialgeometry import Cylinder, Sphere, Cuboid, Mesh

MESH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "41013_A2_CAK",
    "Lynxmotion_Meshes"
)


BRACKET_LIFT = 0.0318
BASE_Z = 0.0329

def safe_mesh_path(filename):
    src = os.path.join(MESH_DIR, filename)
    dst_dir = os.path.join(tempfile.gettempdir(), "lynx_meshes")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, filename)
    shutil.copy(src, dst)
    return dst

# ------------------------------------------------------------
# LINK DIMENSIONS
# ------------------------------------------------------------
D1 = 0.180 # base to shoulder joint
A2 = 0.370 # shoulder joint to elbow
A3 = 0.330 # elbow to forearm
D6 = 0.200 # forearm to centre-point
CLAMP_X = 0.073
CLAMP_Y = 0.08428666
CLAMP_Z = 0.03000001

QLIM_WAIST = [-2 * np.pi, 2 * np.pi]
QLIM_ARM = [-np.pi, np.pi]

# ------------------------------------------------------------
# ACTUATOR MESHES 
# ------------------------------------------------------------
ACT = {
    "m1": dict(file="lss_m1_lo.stl", zc=0.064, xmin=-0.062, xmax=0.075, radius=0.062),
    "s1": dict(file="lss_s1_lo.stl", zc=0.033, xmin=-0.042, xmax=0.050, radius=0.042),
    "l1": dict(file="lss_l1_lo.stl", zc=0.041, xmin=-0.042, xmax=0.050, radius=0.042),
}

def actuator_pose(kind, axis, strip, along=0.0, at=(0, 0, 0)):
    """
    Pose that puts an actuator mesh on a joint.
      axis  : link-frame direction the rotation axis should point
              (the output plate faces this way)
      strip : link-frame direction for the connector strip (must be
              perpendicular to axis)
      along : which point of the actuator sits at 'at':
              -1 = back end, 0 = middle of the body, +1 = output plate
      at    : link-frame position for that point (on the rotation axis)
    """
    a = ACT[kind]
    half = (a["xmax"] - a["xmin"]) / 2
    px = (a["xmax"] + a["xmin"]) / 2 + along * half
    x = np.array(axis, float);  x /= np.linalg.norm(x)
    z = np.array(strip, float); z /= np.linalg.norm(z)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    return SE3(*at) * SE3.Rt(R, [0, 0, 0]) * SE3(-px, 0, -a["zc"])

class LynxmotionSESPro(ERobot):

    def __init__(self):

        l1 = Link(ET.Rz(qlim=QLIM_WAIST), name="waist")
        l2 = Link(ET.tz(D1) * ET.Ry(qlim=QLIM_ARM), name="shoulder", parent=l1)
        l3 = Link(ET.tx(A2) * ET.Ry(qlim=QLIM_ARM), name="elbow", parent=l2)
        l4 = Link(ET.tx(A3) * ET.Rx(qlim=QLIM_ARM), name="wrist1", parent=l3)
        l5 = Link(ET.Ry(qlim=QLIM_ARM), name="wrist2", parent=l4)
        l6 = Link(ET.Rz(qlim=QLIM_ARM), name="tool_roll", parent=l5)

        super().__init__(
            [l1, l2, l3, l4, l5, l6],
            name="LynxmotionSESPro",
            manufacturer="Lynxmotion",
            tool=SE3.Tz(D6),
        )

        self.base = SE3.Tz(BASE_Z)
        deg = np.pi / 180
        self.qr = np.array([0, -60 * deg, 60 * deg, 0, 30 * deg, 0])
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_visuals(self):

        # Colours matched to the real arm: matte black carbon-fibre tubes,
        # slightly lighter charcoal actuator housings, aluminium clamps.
        HOUSING = (0.24, 0.24, 0.26, 1)
        TUBE_COL = (0.13, 0.13, 0.14, 1)
        ALU = (0.65, 0.67, 0.70, 1)
        GRIP = (0.75, 0.75, 0.75, 1)

        base_plate = Mesh(filename=safe_mesh_path("base_bracket.stl"),
                          scale=[1, 1, 1], color=(0.08, 0.08, 0.09, 1),
                          pose=SE3(0, 0, BRACKET_LIFT))

        SERVO_A2, CAP_A2 = 0.06, 0.03
        SERVO_A3, CAP_A3 = 0.03, 0.02
        TUBE_HALF = 0.14375
        TUBE_AXIS = 0.042

        def actuator(kind):
            return Mesh(filename=safe_mesh_path(ACT[kind]["file"]),
                        scale=[1, 1, 1], color=HOUSING)

        BACK = (-1, 0, 0)   # connector strips face backwards on the arm

        parts = [
            # waist: Mega actuator, axis up. Output plate sits just under the
            # shoulder actuator; the body sinks into the base bracket.
            (0, actuator("m1"),
             actuator_pose("m1", axis=(0, 0, 1), strip=BACK, along=1,
                           at=(0, 0, D1 - ACT["m1"]["radius"]))),

            # shoulder: Mega actuator on the Y axis + upper-arm tube + clamp
            (1, actuator("m1"),
             actuator_pose("m1", axis=(0, 1, 0), strip=BACK, along=0)),
            (1, Mesh(filename=safe_mesh_path("cf_tube_280.stl"),
                     scale=[1, 1, 1], color=TUBE_COL),
             SE3(SERVO_A2 + TUBE_HALF, 0, TUBE_AXIS) * SE3.Ry(np.pi / 2)),
            (1, Mesh(filename=safe_mesh_path("cf_clamp.stl"),
                     scale=[1, 1, 1], color=ALU),
             SE3(A2 - CAP_A2, 0, TUBE_AXIS) * SE3.Ry(np.pi / 2)),

            # elbow: Standard actuator on the Y axis + forearm tube + clamp
            (2, actuator("s1"),
             actuator_pose("s1", axis=(0, 1, 0), strip=BACK, along=0)),
            (2, Mesh(filename=safe_mesh_path("cf_tube_280.stl"),
                     scale=[1, 1, 1], color=TUBE_COL),
             SE3(SERVO_A3 + TUBE_HALF, 0, TUBE_AXIS) * SE3.Ry(np.pi / 2)),
            (2, Mesh(filename=safe_mesh_path("cf_clamp.stl"),
                     scale=[1, 1, 1], color=ALU),
             SE3(A3 - CAP_A3, 0, TUBE_AXIS) * SE3.Ry(np.pi / 2)),

            # wrist 1: Standard actuator on the X axis
            (3, actuator("s1"),
             actuator_pose("s1", axis=(1, 0, 0), strip=(0, 0, 1), along=0)),
            # wrist 2: Lite actuator on the Y axis
            (4, actuator("l1"),
             actuator_pose("l1", axis=(0, 1, 0), strip=(0, 0, 1), along=0)),
            # tool roll: Lite actuator on the Z axis, body running forward
            # from the wrist point so it doesn't cross the other two
            (5, actuator("l1"),
             actuator_pose("l1", axis=(0, 0, 1), strip=BACK, along=-1)),

            # gripper stub + fingers, starting where the tool actuator ends
            (5, Cylinder(radius=0.02, length=0.06, color=GRIP),
             SE3(0, 0, 0.092 + 0.03)),
            (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
             SE3(0.02, 0, D6 - 0.025)),
            (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
             SE3(-0.02, 0, D6 - 0.025)),
            ]
        return base_plate, parts

    def pose_visuals(self, q, parts):
        link_frames = self.fkine_all(q)
        for link_idk, shape, offset in parts:
            shape.T = (link_frames[link_idk + 1] * offset).A

if __name__ == "__main__":
    from swift import Swift

    robot = LynxmotionSESPro()
    print(robot)

    base_plate, parts = robot.link_visuals()

    env = Swift()
    env.launch(realtime=True)
    env.add(base_plate)
    for _, shape, _ in parts:
        env.add(shape)

    robot.q = robot.qz
    robot.pose_visuals(robot.q, parts)
    env.step()

    waypoints = [
        robot.qz,
        robot.qr,
        np.array([90, -40, 50, 0, 40, 0]) * np.pi / 180,
        np.array([-90, -50, 70, 20, -30, 45]) * np.pi / 180,
        robot.qr,
    ]

    for i in range(len(waypoints) - 1):
        traj = rtb.jtraj(waypoints[i], waypoints[i + 1], 50)
        for q in traj.q:
            robot.q = q
            robot.pose_visuals(q, parts)
            env.step(0.02)

    env.hold()