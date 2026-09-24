import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import Link, ET, ERobot
from spatialmath import SE3
from spatialgeometry import Cylinder, Sphere

# ------------------------------------------------------------
# LINK DIMENSIONS
# ------------------------------------------------------------
D1 = 0.180 # base to shoulder joint
A2 = 0.450 # shoulder joint to elbow
A3 = 0.370 # elbow to forearm
D6 = 0.080 # forearm to centre-point

QLIM_WAIST = [-2 * np.pi, 2 * np.pi]
QLIM_ARM = [-np.pi, np.pi]

class LynxmotionSESPro(ERobot):

    def __init__(self):

        l1 = Link(ET.Rz(qlim=QLIM_WAIST), name="wasit")
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

        deg = np.pi / 180
        self.qr = np.array([0, -60 * deg, 60 * deg, 0, 30 * deg, 0])
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_visuals(self):

        base_plate = Cylinder(radius=0.09, length=0.02,
                              pose=SE3(0, 0, -0.01), color=(0.1, 0.1, 0.1, 1))
        parts = [
            (0, Cylinder(radius=0.045, length=D1, color=(0.15, 0.15, 0.15, 1)),
                SE3(0, 0, D1 / 2)),
            (1, Cylinder(radius=0.035, length=A2, color=(0.85, 0.1, 0.1, 1)),
                SE3(A2 / 2, 0, 0) * SE3.Ry(np.pi / 2)),
            (2, Cylinder(radius=0.030, length=A3, color=(0.85, 0.1, 0.1, 1)),
                SE3(A3 / 2, 0, 0) * SE3.Ry(np.pi / 2)),
            (3, Sphere(radius=0.035, color=(0.2, 0.2, 0.2, 1)), SE3()),
            (4, Sphere(radius=0.030, color=(0.2, 0.2, 0.2, 1)), SE3()),
            (5, Cylinder(radius=0.025, length=D6, color=(0.7, 0.7, 0.7, 1)),
                SE3(0, 0, D6 / 2)),
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