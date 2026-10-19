##  @file
#   @brief FAIRINO FR3 Robot defined by standard DH parameters with 3D model
#   @date Oct 15, 2026

import swift
import roboticstoolbox as rtb
import spatialmath.base as spb
from spatialmath import SE3
from ir_support.robots.UTSMeshRobot import UTSMeshRobot
import time
import os

from math import pi


# -----------------------------------------------------------------------------------#
class FairinoFR3(UTSMeshRobot):

    def __init__(self, base=None):
        # DH links
        links = self._create_DH()

        # Names of the robot link files in the mesh directory
        link3D_names = dict(
            link0='FairinoFR3Link0',
            link1='FairinoFR3Link1',
            link2='FairinoFR3Link2',
            link3='FairinoFR3Link3',
            link4='FairinoFR3Link4',
            link5='FairinoFR3Link5',
            link6='FairinoFR3Link6'
        )

        # A joint config and the 3D object transforms to match that config
        qtest = [0, -pi/2, 0, -pi/2, 0, 0]

        qtest_transforms = [
            # Link0
            spb.transl(0, 0, 0),

            # Link1
            spb.transl(0, 0, 0),

            # Link2
            spb.transl(0, 0, 0),

            # Link3
            # spb.transl(-0.001, -0.003, 0.558) @
            # spb.rpy2tr(pi/2, -pi/2, 0, order='xyz'),
            spb.transl(0, 0, 0),

            # Link4 - not calibrated yet
            spb.transl(0, 0, 0),

            # Link5 - not calibrated yet
            spb.transl(0, 0, 0),

            # Link6 - not calibrated yet
            spb.transl(0, 0, 0)
        ]

        current_path = os.path.abspath(os.path.dirname(__file__))
        mesh_path = os.path.join(current_path, "FAIRINO FR3_Meshes")

        super().__init__(
            links=links,
            mesh_stem="FairinoFR3",
            mesh_dir=mesh_path,
            name="FairinoFR3",
            home_q=qtest,
            base=base,
            link3d_names=link3D_names,
            qtest_transforms=qtest_transforms,
        )

    # -----------------------------------------------------------------------------------#
    def _create_DH(self):
        a = [0, -0.280, -0.24001, 0, 0, 0]
        d = [0.140, 0, 0, -0.102, 0.102, 0.100]
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]

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
            link = rtb.RevoluteDH(
                d=d[i],
                a=a[i],
                alpha=alpha[i],
                qlim=qlim[i]
            )
            links.append(link)

        return links

    # -----------------------------------------------------------------------------------#
    def test(self):
        env = swift.Swift()
        env.launch(realtime=True)

        # Start at qtest
        self.q = self._qtest

        # Offset whole robot in Swift like the tutor example
        self.base = SE3(0.5, 0.5, 0)

        self.add_to_env(env)

        # Simple trajectory
        q_goal = [self.q[i] - pi/6 for i in range(self.n)]
        qtraj = rtb.jtraj(self.q, q_goal, 50).q

        #q_goal = self.q.copy()

        # # Move J3 only
        # q_goal[3] += pi / 6
        # q_goal[4] += pi / 6   # J5

        # qtraj = rtb.jtraj(self.q, q_goal, 50).q

        for q in qtraj:
            self.q = q
            env.step(0.02)

        time.sleep(3)
        env.hold()


# ---------------------------------------------------------------------------------------#
if __name__ == "__main__":
    r = FairinoFR3()
    #r.test()
