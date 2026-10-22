"""
FAIRINO FR3 ROBOT MODEL
=======================

Defines the FAIRINO FR3 as a six degree-of-freedom serial manipulator using
standard Denavit-Hartenberg (DH) parameters.
--------------
This file demonstrates that the robot is parameter-based rather than simply
an imported animation.

The Robotics Toolbox DH model defines the actual robot kinematics used for:
    - forward kinematics
    - inverse kinematics
    - Jacobian calculation
    - RMRC
    - trajectory planning
    - joint-limit checking

The DAE meshes are attached to those mathematical links only for
visualisation in Swift.
"""

import os
from math import pi

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.UTSMeshRobot import UTSMeshRobot
import time


# -----------------------------------------------------------------------------------#
class FairinoFR3(UTSMeshRobot):
    """
        Six-axis FAIRINO FR3 collaborative robot model.

        The model combines:
            1. a mathematical DHRobot model used for kinematics/planning
            2. link meshes used for 3D visualisation in Swift

        Keeping these together means the visual robot follows the same joint
        configuration used by the Robotics Toolbox calculations.
        """

    def __init__(self, base=None):

        # Create the mathematical six-joint robot model.
        links = self._create_DH()

        # Each DH link is associated with its corresponding exported mesh.
        # These meshes do NOT define the kinematics; they only visualise the
        # link poses produced by the DH model.

        link3D_names = dict(
            link0='FairinoFR3Link0',
            link1='FairinoFR3Link1',
            link2='FairinoFR3Link2',
            link3='FairinoFR3Link3',
            link4='FairinoFR3Link4',
            link5='FairinoFR3Link5',
            link6='FairinoFR3Link6'
        )

        # Validated starting configuration.
        # This configuration places the robot in a useful position
        # instead of starting every joint at zero.
        q_home = [0, -pi/2, 0, -pi/2, 0, 0]

        # The link meshes were aligned to their DH link frames during model
        # development, so no additional link-specific visual corrections are
        # required here.

        qtest_transforms = [
            # Link0
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0),
            spb.transl(0, 0, 0)
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_path = os.path.join(current_path, "FAIRINO FR3_Meshes")

        super().__init__(
            links=links,
            mesh_stem="FairinoFR3",
            mesh_dir=mesh_path,
            name="FairinoFR3",
            home_q=q_home,
            base=base,
            link3d_names=link3D_names,
            qtest_transforms=qtest_transforms,
        )

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):

        """
        Construct the FAIRINO FR3 from standard DH parameters.

        For each revolute joint:
            theta = current joint position
            d     = link offset
            a     = link length
            alpha = link twist

        These parameters determine the transform between consecutive
        coordinate frames and therefore define the robot's forward
        kinematics.

        Joint limits are included so IK, RMRC and teach/jog operation use
        physically meaningful ranges rather than unrestricted joint angles.
        """

        # Link lengths
        a = [0, -0.280, -0.24001, 0, 0, 0]
        # Link offsets
        d = [0.140, 0, 0, -0.102, 0.102, 0.100]
        # Joint twists
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

        # Physical joint limits converted from degrees to rad
        qlim = [
            [-175*pi/180, 175*pi/180],
            [-265*pi/180,  85*pi/180],
            [-162*pi/180, 162*pi/180],
            [-265*pi/180,  85*pi/180],
            [-175*pi/180, 175*pi/180],
            [-175*pi/180, 175*pi/180]
        ]

        links = []
        for i in range(6):
            link = rtb.RevoluteDH(d=d[i], a=a[i], alpha=alpha[i], qlim=qlim[i])
            links.append(link)

        return links

    # -----------------------------------------------------------------------------------#
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)

        # Start at qtest now finalised to be q_home
        self.q = self._qtest

        # Offset whole robot in Swift like in example
        self.base = SE3(0.5, 0.5, 0)

        self.add_to_env(env)

        # Simple trajectory
        q_goal = [self.q[i] - pi/6 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q

        for q in qtraj:
            self.q = q
            env.step(0.02)

        time.sleep(3)
        env.hold()


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = FairinoFR3()
    #r.test()
