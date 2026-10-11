import sys
import numpy as np
import roboticstoolbox as rtb

from lynxmotion_ses_pro_model import LynxmotionSESPro

DEG = np.pi / 180   # convert from radians to degrees
STEPS = 60          # frames between waypoints
FRAME_DT = 0.02     # seconds/frame

def make_waypoints(robot):
    return [
        robot.qz,
        robot.qr,
        np.array([60, -20, 50, 90, -30, 90])  * DEG,    # waist swing + wrist roll/yaw
        np.array([-60, 20, 70, -60, 45, -90]) * DEG,    # reach the other side
        np.array([0, -50, 100, 0, 60, 180])   * DEG,    # fold in, spin the gripper
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