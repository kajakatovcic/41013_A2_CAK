# Testing modelling DH parameters using CircularDH robot before creating a more complex mesh.
from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from ir_support import CylindricalDHRobotPlot
import swift

deg = pi/180

links = [RevoluteDH(d=0.2522, a=0, alpha=-pi/2, qlim=[-179*deg, 179*deg]),
         RevoluteDH(d=0, a=0.24152, alpha=0, offset=-pi/2, qlim=[-80*deg, 140*deg]),
         RevoluteDH(d=0, a=0, alpha=pi/2, offset=+pi/2, qlim=[-80*deg, 140*deg]),
         RevoluteDH(d=0.3, a=0, alpha=-pi/2, qlim=[-179*deg, 179*deg]),
         RevoluteDH(d=0, a=0, alpha=pi/2, qlim=[-95*deg, 95*deg]),
         RevoluteDH(d=0.129, a=0, alpha=0, qlim=[-179*deg, 179*deg])
]
robot = DHRobot(links, name="Igus_ReBeL")
qr = np.array([0, 45, 45, 0, 90, 0]) * deg # ready pose with tool pointing down
qh = np.array([0, 90, 0, 0, 0, 0]) * deg # should be horizontal with flange 0.670m out at 0.252m height
robot.q = qr

cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="blue")
robot = cyl_viz.create_cylinders()

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)
env.hold()