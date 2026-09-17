# Testing modelling DH parameters using cylindricalDHRobotPlot before creating a more complex mesh.
from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from ir_support import CylindricalDHRobotPlot
import swift

deg = pi/180

links = [RevoluteDH(d=0.187, a=0, alpha=pi/2, qlim=[-2*pi, 2*pi]), 
         RevoluteDH(d=0, a=0.21, alpha=0, offset=+pi/2, qlim=[-120*deg, 120*deg]),
         RevoluteDH(d=0.006, a=0, alpha=-pi/2, offset=-pi/2, qlim=[-125*deg, 125*deg]),
         RevoluteDH(d=0.2105, a=0, alpha=pi/2, qlim=[-2*pi, 2*pi]),
         RevoluteDH(d=0, a=0, alpha=-pi/2, qlim=[-120*deg, 120*deg]),
         RevoluteDH(d=0.1593, a=0, alpha=0, qlim=[-2*pi, 2*pi])
]
robot = DHRobot(links, name="JAKA_MiniCobo")
qr = np.array([0, -30, -60, 0, -90, 0]) * deg # ready pose with tool pointing down
qh = np.array([0, 0, 0, 0, 0, 0]) # all zeros
robot.q = qr

cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="blue")
robot = cyl_viz.create_cylinders()

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)
env.hold()