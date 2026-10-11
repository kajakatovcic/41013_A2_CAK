"""
Lynxmotion SES-PRO 900mm 6DoF arm - MOVEMENT / demo script.

What the robot DOES lives here (poses, waypoints, animation in Swift).
What the robot IS (meshes, link offsets, actuator placement, joint limits) lives in
lynxmotion_model.py, which must sit in the same folder as this file.

    python3 "Lynxmotion SES-PRO.py"          # animate through the waypoints
    python3 "Lynxmotion SES-PRO.py" static   # just show the photo pose (qr)
"""
import sys
import numpy as np
import roboticstoolbox as rtb

from lynxmotion_model import LynxmotionSESPro

DEG = np.pi / 180
STEPS = 60          # frames between waypoints
FRAME_DT = 0.02     # seconds per frame


def make_waypoints(robot):
    """Joint-space waypoints [q0..q5] in radians: waist, shoulder, elbow, wrist roll, pitch, yaw."""
    return [
        robot.qz,
        robot.qr,
        np.array([60, -20, 50, 90, -30, 90]) * DEG,      # waist swing + wrist roll/yaw
        np.array([-60, 20, 70, -60, 45, -90]) * DEG,     # reach the other side
        np.array([0, -50, 100, 0, 60, 180]) * DEG,       # fold in, spin the gripper
        robot.qr,
    ]


def animate(robot, env, parts, waypoints):
    for i in range(len(waypoints) - 1):
        traj = rtb.jtraj(waypoints[i], waypoints[i + 1], STEPS)
        for q in traj.q:
            robot.q = q
            robot.pose_visuals(q, parts)
            env.step(FRAME_DT)


def main():
    from swift import Swift

    robot = LynxmotionSESPro()
    print(robot)

    base_plate, parts = robot.link_visuals()

    env = Swift()
    env.launch(realtime=True)
    env.add(base_plate)
    for _, shape, _ in parts:
        env.add(shape)

    # "static" -> just show the photo pose so you can compare it with the picture
    if "static" in sys.argv[1:]:
        robot.q = robot.qr
        robot.pose_visuals(robot.q, parts)
        env.step()
        env.hold()
        return

    robot.q = robot.qz
    robot.pose_visuals(robot.q, parts)
    env.step()

    animate(robot, env, parts, make_waypoints(robot))
    env.hold()


if __name__ == "__main__":
    main()