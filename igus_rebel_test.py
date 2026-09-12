# Testing modelling DH parameters using CircularDH robot before creating a more complex mesh.
from math import pi
from roboticstoolbox import DHRobot, RevoluteDH
from ir_support import CylindricalDHRobotPlot
import swift


links = [RevoluteDH(d=0.2522, a=0, alpha=-pi/2),
         RevoluteDH(d=0, a=0.24152, alpha=0, offset=-pi/2),
         RevoluteDH(d=0, a=0, alpha=pi/2, offset=+pi/2),
         RevoluteDH(d=0.3, a=0, alpha=-pi/2),
         RevoluteDH(d=0, a=0, alpha=pi/2),
         RevoluteDH(d=0.129, a=0, alpha=0)
]
robot = DHRobot(links, name="Igus_ReBeL")
robot.q = [0,0,0,0,0,0]

cyl_viz = CylindricalDHRobotPlot(robot, cylinder_radius=0.05, color="blue")
robot = cyl_viz.create_cylinders()

env = swift.Swift()
env.launch(realtime=True)
env.add(robot)