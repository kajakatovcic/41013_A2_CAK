import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import Link, ET, ERobot
from spatialmath import SE3
from spatialgeometry import Cylinder, Sphere, Cuboid

# ------------------------------------------------------------
# LINK DIMENSIONS
# ------------------------------------------------------------
D1 = 0.180 # base to shoulder joint
A2 = 0.370 # shoulder joint to elbow
A3 = 0.330 # elbow to forearm
D6 = 0.200 # forearm to centre-point

QLIM_WAIST = [-2 * np.pi, 2 * np.pi]
QLIM_ARM = [-np.pi, np.pi]

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

        deg = np.pi / 180
        self.qr = np.array([0, -60 * deg, 60 * deg, 0, 30 * deg, 0])
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_visuals(self):

        DARK = (0.12, 0.12, 0.12, 1)
        CAP = (0.32, 0.32, 0.32, 1)
        RED = (0.85, 0.1, 0.1, 1)
        GRIP = (0.75, 0.75, 0.75, 1)

        base_plate = Cylinder(radius=0.09, length=0.02,
                              pose=SE3(0, 0, -0.01), color=(0.1, 0.1, 0.1, 1))

        WAIST_BOX = 0.09
        post_len = D1 - WAIST_BOX
        SERVO_A2, CAP_A2 = 0.06, 0.03
        TUBE_A2 = A2 - SERVO_A2 - CAP_A2
        SERVO_A3, CAP_A3 = 0.03, 0.02
        TUBE_A3 = A3 - SERVO_A3 - CAP_A3

        parts = [
            (0, Cylinder(radius=0.035, length=post_len, color=DARK),
             SE3(0, 0, post_len / 2)),
            (0, Cuboid(scale=[0.08, 0.07, WAIST_BOX],color=DARK),
             SE3(0, 0, post_len + WAIST_BOX / 2)),

            (1, Cuboid(scale=[SERVO_A2, 0.06, 0.06], color=DARK),
             SE3(SERVO_A2 / 2, 0, 0)),
            (1, Cylinder(radius=0.025, length=TUBE_A2, color=RED),
             SE3(SERVO_A2 + TUBE_A2 / 2, 0, 0) * SE3.Ry(np.pi / 2)),
            (1, Cuboid(scale=[CAP_A2, 0.05, 0.05], color=CAP),
             SE3(A2 - CAP_A2 / 2, 0, 0)),

            (2, Cuboid(scale=[SERVO_A3, 0.05, 0.05], color=DARK),
             SE3(SERVO_A3 / 2, 0, 0)),
            (2, Cylinder(radius=0.020, length=TUBE_A3, color=RED),
             SE3(SERVO_A3 + TUBE_A3 / 2, 0, 0) * SE3.Ry(np.pi / 2)),
            (2, Cuboid(scale=[CAP_A3, 0.04, 0.04], color=CAP),
             SE3(A3 - CAP_A3 / 2, 0, 0)),  

            (3, Cuboid(scale=[0.05, 0.05, 0.05], color=DARK), SE3()),
            (4, Cuboid(scale=[0.045, 0.045, 0.045], color=DARK), SE3()),  

            (5, Cuboid(scale=[0.04, 0.04, 0.04], color=DARK), SE3()), 
            (5, Cylinder(radius=0.018, length=D6 * 0.6, color=GRIP),
             SE3(0, 0, D6 * 0.3)),
            (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
             SE3(0.02, 0, D6 * 0.75)),
            (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
             SE3(-0.02, 0, D6 * 0.75)),     
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