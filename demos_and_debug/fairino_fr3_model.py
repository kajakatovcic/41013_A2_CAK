"""
FAIRINO FR3 - standard DH model for Robotics Toolbox / Swift.

This model intended for initial testing and modelling of the FAIRNO FR3 Robot:
1. Verify the kinematics with CylindricalDHRobotPlot.
2. Test candidate joint configurations.
3. Only after the DH model is correct, attach/export detailed meshes.

Dimensions  in metres.

FR3 kinematic dimensions used here:
    d1 = 0.140
    a2 = -0.280
    a3 = -0.24001
    d4 = 0.102
    d5 = 0.102
    d6 = 0.100

The signs on a2/a3 follow the frame convention used by the official FAIRINO
FR3 V6 URDF chain. 

Joint limits below follow the supplied FR3 mechanical drawing / FAIRINO FR3
specification.

NOTE: I HAD SO MANY ISSUES WITH GIT SO I WAS WORKING ON AN INDIVIDUAL LOCAL FILE SO MY GITCOMMITS ARE NOT CONSISTENT 
THROUGH THE WEEKS BUT I SWEAR I HAVE BEEN WORKING ON THIS ASSIGNMENT :,(
"""

from math import pi
import numpy as np
import swift

from roboticstoolbox import DHRobot, RevoluteDH
from ir_support import CylindricalDHRobotPlot


DEG = pi / 180.0


def create_fairino_fr3():
    """
    Create the FAIRINO FR3 using standard DH parameters.

    Returns
    -------
    DHRobot
        Robotics Toolbox serial-link robot.
    """

    # Mechanical dimensions [m]
    d1 = 0.140
    a2 = -0.280
    a3 = -0.24001
    d4 = 0.102
    d5 = 0.102
    d6 = 0.100

    links = [
        # Joint 1 - base
        RevoluteDH(d=d1,a=0,alpha=pi / 2,qlim=[-175 * DEG, 175 * DEG],),

        # Joint 2 - shoulder
        RevoluteDH(d=0,a=a2,alpha=0,qlim=[-265 * DEG, 85 * DEG],),

        # Joint 3 - elbow
        RevoluteDH(d=0,a=a3,alpha=0,qlim=[-150 * DEG, 150 * DEG],),

        # Joint 4 - wrist 1
        RevoluteDH(d=d4, a=0,alpha=pi / 2,qlim=[-265 * DEG, 85 * DEG],),

        # Joint 5 - wrist 2
        RevoluteDH(d=d5,a=0,alpha=-pi / 2,qlim=[-175 * DEG, 175 * DEG],),

        # Joint 6 - wrist 3 / flange
        RevoluteDH(d=d6,a=0,alpha=0,qlim=[-175 * DEG, 175 * DEG],),
    ]

    robot = DHRobot(links, name="FAIRINO_FR3")

    # Useful starting posture for visual inspection.
    # This is only a test simulation pose, not certified safe pose.
    robot.q = np.deg2rad([0, -90, 90, -90, -90, 0])

    return robot


def print_dh_table(robot):
    print("\nFAIRINO FR3 standard DH table")
    print("Joint |      d (m) |      a (m) | alpha (deg) | q min/max (deg)")
    print("-" * 72)

    for i, link in enumerate(robot.links, start=1):
        qlim_deg = np.degrees(link.qlim)
        print(
            f"{i:5d} | "
            f"{link.d:10.5f} | "
            f"{link.a:10.5f} | "
            f"{np.degrees(link.alpha):11.1f} | "
            f"{qlim_deg[0]:7.1f} to {qlim_deg[1]:7.1f}"
        )


def run_cylindrical_test():
    """
    Open the FR3 in Swift using the simple cylindrical representation.
    This should be the first verification step before adding Blender meshes.
    """

    robot = create_fairino_fr3()
    print_dh_table(robot)

    cyl_viz = CylindricalDHRobotPlot(robot,cylinder_radius=0.035,color="blue",)
    robot = cyl_viz.create_cylinders()

    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)
    env.step()

    print("\nStarting joint state [deg]:")
    print(np.round(np.degrees(robot.q), 2))

    print("\nStarting FK:")
    print(robot.fkine(robot.q))

    input(
        "\nInspect the FR3 link directions and joint axes in Swift.\n"
        "Press Enter to close/finish.\n"
    )


if __name__ == "__main__":
    run_cylindrical_test()
