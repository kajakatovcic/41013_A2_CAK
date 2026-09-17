import os
from math import pi
import roboticstoolbox as rtb
from ir_support.robots.UTSMeshRobot import UTSMeshRobot

deg = pi / 180

MESH_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "JAKA_Meshes")


class JakaMiniCobo(UTSMeshRobot):
    """
    JAKA MiniCobo: 6-DOF collaborative arm with a spherical wrist.
    """
 
    def __init__(self, base=None):
        links = self._create_DH()
        super().__init__(
            links=links,
            mesh_stem="JakaMiniCobo",
            mesh_dir=MESH_DIR,
            name="JakaMiniCobo",
            home_q=[0, 0, 0, 0, 0, 0],
            base=base,
        )
 
    def _create_DH(self):
        return [
            rtb.RevoluteDH(d=0.187, a=0, alpha=pi / 2, qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0, a=0.21, alpha=0, offset=+pi / 2, qlim=[-120 * deg, 120 * deg]),
            rtb.RevoluteDH(d=0.006,  a=0, alpha=-pi/ 2, offset=-pi / 2, qlim=[-125 * deg, 125 * deg]),
            rtb.RevoluteDH(d=0.2105, a=0, alpha=pi/2, qlim=[-2 * pi, 2 * pi]),
            rtb.RevoluteDH(d=0, a=0, alpha=-pi/2, qlim=[-120 * deg, 120 * deg]),
            rtb.RevoluteDH(d=0.1593, a=0, alpha=0, qlim=[-2 * pi, 2 * pi]),
        ]
 
 
if __name__ == "__main__":
    import numpy as np
    import swift
    robot = JakaMiniCobo()
    env = swift.Swift()
    env.launch(realtime=True)
    robot.add_to_env(env)
    env.hold()