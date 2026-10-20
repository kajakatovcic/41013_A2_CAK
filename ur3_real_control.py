#!/usr/bin/env python3
# Roslibpy script for controlling the real UR3e.
# Sourced from base 'new_ur_control_script-1.py' from IR Canvas page
# Customised to suit the UR3e's purpose in our workcell, i.e. pick and place
# action of plates into a designated 'drop-off' zone within its workspace.
import time
import roslibpy
import numpy as np

current_pos = None

JOINT_NAMES = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint'
]

# UR3e DH lengths (m)
D1, A2, A3, D5, D6 = 0.15185, 0.24355, 0.2132, 0.08535, 0.0921

def tool_down_q(reach, z, pan=0.0):
    """Elbow-up joint config with the tool pointing straight down.

    reach: horizontal distance (m) from the base axis to the flange, in the arm's plane
    z: flange height (m) above the base. This is the flange, NOT the gripper tip.
    Tool points down when wrist2 = -pi/2 and lift + elbow + wrist1 = -pi/2.
    """
    # Wrist 1 centre sits D5 back from the flange and D6 above it. Solves the 2-link arm to it
    r = reach - D5
    h = z + D6 - D1
    c = (r**2 + h**2 - A2**2 - A3**2) / (2 * A2 * A3)
    if abs(c) > 1.0:
        raise ValueError(f"Tool-down pose out of reach: reach={reach}, z={z}")
    elbow = np.arccos(c)
    lift = -(np.arctan2(h, r) + np.arctan2(A3 * np.sin(elbow), A2 + A3 * np.cos(elbow)))
    wrist1 = -np.pi/2 - lift - elbow
    return [pan, lift, elbow, wrist1, -np.pi/2, 0.0]

# Pick location and heights (flange). Raise Z_RAISE if the gripper still gets too close to the plate/base surface.
PICK_REACH = 0.2986
Z_RAISE = 0.10  # extra clearance for the long gripper

# Joint configs [pan, lift, elbow, wrist1, wrist2, wrist3] in radians
HOME_Q     = [0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0]          # arm straight up
DOWN_Q     = tool_down_q(PICK_REACH, 0.30 + Z_RAISE)           # tool down, high
APPROACH_Q = tool_down_q(PICK_REACH, 0.20 + Z_RAISE)           # tool down, above object
PICK_Q     = tool_down_q(PICK_REACH, 0.15 + Z_RAISE)           # tool down, grasp height

# Drop-off: same poses with the base turned 90 deg (+ = counterclockwise seen from above).
# Only the pan joint changes, so the tool stays pointing down at the same reach and height.
DROP_PAN = np.pi/2

def rotate_base(q, pan_offset):
    """Return a copy of joint config q with the shoulder pan joint offset by pan_offset."""
    return [q[0] + pan_offset] + list(q[1:])

DROP_APPROACH_Q = rotate_base(APPROACH_Q, DROP_PAN)  # tool down, same height as APPROACH_Q
DROP_Q          = rotate_base(PICK_Q, DROP_PAN)      # tool down, same height as PICK_Q

def joint_state_cb(message):
    global current_pos
    # /joint_states is published in alphabetical order, so reorder by name
    name_to_pos = dict(zip(message['name'], message['position']))
    current_pos = [name_to_pos[n] for n in JOINT_NAMES]

def move_ur_joint_positions(client, joint_positions, duration=5.0):
    global current_pos
    # Clear the previous reading so each move starts from a fresh joint state
    current_pos = None


    try:
        client.run()

        # Subscribe to joint states to get the current position
        listener = roslibpy.Topic(client, '/ur/joint_states', 'sensor_msgs/JointState')
        listener.subscribe(joint_state_cb)

        # Wait until we receive a joint state
        print("[ROS] Waiting for current joint state...")
        start_time = time.time()
        while current_pos is None and time.time() - start_time < 5.0:
            time.sleep(0.05)
        if current_pos is None:
            raise RuntimeError("No joint state received from /ur/joint_states")

        print(f"[ROS] Current joint positions: {current_pos}")

        # Build a JointTrajectory message for the scaled_pos_joint_traj_controller
        trajectory_msg = {
            'joint_names': JOINT_NAMES,
            'points': [
                {
                    'positions': current_pos,
                    'time_from_start': {'secs': 0, 'nsecs': 0}
                },
                {
                    'positions': joint_positions,
                    'time_from_start': {
                        'secs': int(duration),
                        'nsecs': int((duration - int(duration)) * 1e9)
                    }
                }
            ]
        }

        # Publish to the controller's /command topic
        topic = roslibpy.Topic(
            client,
            '/ur/scaled_pos_joint_traj_controller/command',
            'trajectory_msgs/JointTrajectory'
        )
        topic.advertise()
        topic.publish(roslibpy.Message(trajectory_msg))
        print("[ROS] Trajectory published.")

        # Wait for motion to complete
        time.sleep(duration + 1.0)

        topic.unadvertise()
        listener.unsubscribe()

    finally:
        print("[ROS] Finish")

if __name__ == '__main__':
    client = roslibpy.Ros(host='192.168.27.1', port=9090)  # Replace with your ROS bridge IP
    client.run()
    try:
        # Home -> tool pointing down -> approach -> pick, then back up
        move_ur_joint_positions(client, HOME_Q, duration=7.0)
        move_ur_joint_positions(client, DOWN_Q, duration=7.0)
        move_ur_joint_positions(client, APPROACH_Q, duration=4.0)
        move_ur_joint_positions(client, PICK_Q, duration=3.0)
        move_ur_joint_positions(client, APPROACH_Q, duration=3.0)

        # Turn the base 90 deg at approach height, lower, then lift clear
        move_ur_joint_positions(client, DROP_APPROACH_Q, duration=6.0)
        move_ur_joint_positions(client, DROP_Q, duration=3.0)
        move_ur_joint_positions(client, DROP_APPROACH_Q, duration=3.0)
        move_ur_joint_positions(client, DOWN_Q, duration=7.0)
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user.")
