import time
import numpy as np
import roboticstoolbox as rtb
import swift

from math import pi
from spatialmath import SE3
from spatialgeometry import Cuboid
from fairino_fr3 import FairinoFR3
from fairino_fr3_station import FR3Gripper


def update_gripper(robot, gripper):

    T_flange = robot.fkine(robot.q)

    gripper.update(T_flange)


def move_robot(robot, gripper, env, q_goal, steps=100):

    q_start = robot.q.copy()

    qtraj = rtb.jtraj(
        q_start,
        q_goal,
        steps
    ).q

    for q in qtraj:

        robot.q = q

        update_gripper(robot, gripper)

        env.step(0.03)


def animate_gripper(robot, gripper, env, opening_goal, steps=20):

    opening_start = gripper.opening

    values = np.linspace(
        opening_start,
        opening_goal,
        steps
    )

    for opening in values:

        gripper.opening = opening

        update_gripper(robot, gripper)

        env.step(0.05)


def main():

    env = swift.Swift()
    env.launch(realtime=True)

    robot = FairinoFR3()

    robot.base = SE3(0.5, 0.5, 0)
    robot.q = robot.home_q.copy()

    robot.add_to_env(env)

    # ---------------------------------------------------------
    # Add gripper
    # ---------------------------------------------------------

    gripper = FR3Gripper()

    T_flange = robot.fkine(robot.q)
    gripper.update(T_flange)

    gripper.add_to_env(env)

    env.step(0.1)
    # env.hold()      

# ---------------------------------------------------------
# TEMPORARY FRAME-6 AXIS MARKERS
# ---------------------------------------------------------

    # T_flange = robot.fkine(robot.q)

    # axis_length = 0.12
    # axis_thickness = 0.008

    # x_axis = Cuboid(
    #     scale=[axis_length, axis_thickness, axis_thickness],
    #     color=(1.0, 0.0, 0.0, 1.0)
    # )

    # y_axis = Cuboid(
    #     scale=[axis_thickness, axis_length, axis_thickness],
    #     color=(0.0, 1.0, 0.0, 1.0)
    # )

    # z_axis = Cuboid(
    #     scale=[axis_thickness, axis_thickness, axis_length],
    #     color=(0.0, 0.0, 1.0, 1.0)
    # )

    # x_axis.T = (
    #     T_flange * SE3(axis_length / 2, 0, 0)
    # ).A

    # y_axis.T = (
    #     T_flange * SE3(0, axis_length / 2, 0)
    # ).A

    # z_axis.T = (
    #     T_flange * SE3(0, 0, axis_length / 2)
    # ).A

    # env.add(x_axis)
    # env.add(y_axis)
    # env.add(z_axis)

    # env.step(0.1)

    # env.hold()

    # ---------------------------------------------------------
    # Continue with your normal tests below
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Test 1 - Open / close while stationary
    # ---------------------------------------------------------

    print("Closing gripper")

    animate_gripper(
        robot,
        gripper,
        env,
        FR3Gripper.CLOSED
    )

    time.sleep(1)

    print("Opening gripper")

    animate_gripper(
        robot,
        gripper,
        env,
        FR3Gripper.OPEN
    )

    time.sleep(1)

    # ---------------------------------------------------------
    # Test 2 - Move robot while gripper stays attached
    # ---------------------------------------------------------

    q_goal = robot.q.copy()

    q_goal[0] += pi / 6
    q_goal[1] += pi / 8
    q_goal[4] += pi / 6
    q_goal[5] += pi / 4

    print("Moving robot")

    move_robot(
        robot,
        gripper,
        env,
        q_goal
    )

    time.sleep(1)

    # Return home
    print("Returning home")

    move_robot(
        robot,
        gripper,
        env,
        robot.home_q.copy()
    )

    env.hold()


if __name__ == "__main__":
    main()