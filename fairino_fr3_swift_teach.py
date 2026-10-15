"""
FAIRINO FR3 Swift Teach Mode

Run:
    python fairino_fr3_swift_teach.py

Use the Swift GUI to:
- move J1-J6 independently
- open / close the FR3 gripper
- print or save taught poses
- keep the tray parked at the FR3 station stop
"""

import numpy as np
import swift

from spatialmath import SE3
from spatialmath.base import tr2rpy

from fairino_fr3 import FairinoFR3
from fairino_fr3_station import FR3Station, FR3Gripper
from fairino_fr3_motion import TOOL
from fairino_fr3_task import FR3Job

from workcell_layout import STATIONS, TRAY_ENTRY_X
from workcell_scene import Workcell


ROBOT_FORWARD_OFFSET = 0.15
START_Q_DEG = np.array([0, -90, 90, -90, -90, 0], dtype=float)


class FR3SwiftTeach:

    def __init__(self):
        self.station_base = STATIONS["FAIRINO_FR3"][0]
        self.robot_base = self.station_base * SE3(0, ROBOT_FORWARD_OFFSET, 0)

        self.robot = FairinoFR3(base=self.robot_base)
        self.robot.q = np.deg2rad(START_Q_DEG.copy())

        self.station = FR3Station(self.station_base, self.robot_base)

        self.cell = Workcell(
            tray_x=TRAY_ENTRY_X,
            tray_placeholders=()
        )

        # Same tray stop calculation as the normal FR3 task.
        self.job = FR3Job(self.station, self.cell.conveyor.tray)
        self.cell.conveyor.tray.set_x(self.job.tray_stop_x)

        self.saved_poses = []
        self.gripper_closed = False
        self.env = None

        self.joint_label = None
        self.tcp_label = None
        self.rpy_label = None
        self.gripper_label = None

    def tcp_pose(self):
        return self.robot.fkine(self.robot.q) * TOOL

    def update_visuals(self):
        T_flange = self.robot.fkine(self.robot.q)
        self.station.gripper.update(T_flange)

    def update_labels(self):
        q_deg = np.degrees(self.robot.q)
        T_tcp = self.tcp_pose()
        xyz = T_tcp.t
        rpy = tr2rpy(T_tcp.A, unit="deg")

        self.joint_label.desc = (
            "q [deg]: "
            + np.array2string(np.round(q_deg, 1), separator=", ")
        )
        self.tcp_label.desc = (
            f"TCP xyz [m]: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}"
        )
        self.rpy_label.desc = (
            f"TCP rpy [deg]: {rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}"
        )
        self.gripper_label.desc = (
            "Gripper: CLOSED" if self.gripper_closed else "Gripper: OPEN"
        )

    def joint_slider_callback(self, value, index):
        q_new = np.deg2rad(float(value))
        q_new = np.clip(
            q_new,
            self.robot.qlim[0, index],
            self.robot.qlim[1, index]
        )

        q = np.array(self.robot.q, dtype=float)
        q[index] = q_new
        self.robot.q = q

        self.update_visuals()
        self.update_labels()

    def open_gripper(self, _=None):
        self.station.gripper.opening = FR3Gripper.OPEN
        self.gripper_closed = False
        self.update_visuals()
        self.update_labels()

    def close_gripper(self, _=None):
        self.station.gripper.opening = FR3Gripper.CLOSED
        self.gripper_closed = True
        self.update_visuals()
        self.update_labels()

    def reset_pose(self, _=None):
        self.robot.q = np.deg2rad(START_Q_DEG.copy())
        self.station.gripper.opening = FR3Gripper.OPEN
        self.gripper_closed = False
        self.update_visuals()
        self.update_labels()
        print("\nReset q [deg]:", np.round(np.degrees(self.robot.q), 2))

    def print_pose(self, _=None):
        T_tcp = self.tcp_pose()
        print("\n" + "=" * 72)
        print("CURRENT FAIRINO FR3 TEACH POSE")
        print("=" * 72)
        print("q [deg]:", np.round(np.degrees(self.robot.q), 3))
        print("q [rad]:", np.round(self.robot.q, 6))
        print("TCP xyz [m]:", np.round(T_tcp.t, 4))
        print("TCP rpy [deg]:", np.round(tr2rpy(T_tcp.A, unit="deg"), 2))
        print("Gripper:", "CLOSED" if self.gripper_closed else "OPEN")

    def save_pose(self, _=None):
        n = len(self.saved_poses) + 1
        q = np.array(self.robot.q, dtype=float).copy()
        T_tcp = self.tcp_pose()

        self.saved_poses.append({
            "number": n,
            "q": q,
            "gripper_closed": self.gripper_closed,
            "tcp": T_tcp,
        })

        print("\n" + "#" * 72)
        print(f"SAVED TEACH POSE {n}")
        print("#" * 72)
        print(
            "q_deg = np.deg2rad("
            + repr(np.round(np.degrees(q), 2).tolist())
            + ")"
        )
        print("gripper =", '"closed"' if self.gripper_closed else '"open"')
        print("TCP xyz =", np.round(T_tcp.t, 4).tolist())
        print(
            "TCP rpy deg =",
            np.round(tr2rpy(T_tcp.A, unit="deg"), 2).tolist()
        )
        print(f"Take a screenshot and label it Pose {n}.")

    def add_gui(self):
        self.joint_label = swift.Label("")
        self.tcp_label = swift.Label("")
        self.rpy_label = swift.Label("")
        self.gripper_label = swift.Label("")

        self.env.add(swift.Label("FAIRINO FR3 TEACH MODE"))
        self.env.add(self.joint_label)
        self.env.add(self.tcp_label)
        self.env.add(self.rpy_label)
        self.env.add(self.gripper_label)

        # Lab-style Swift sliders, using the FR3's actual qlim values.
        for i in range(self.robot.n):
            q_min_deg = float(np.degrees(self.robot.qlim[0, i]))
            q_max_deg = float(np.degrees(self.robot.qlim[1, i]))
            q_initial_deg = float(np.degrees(self.robot.q[i]))

            self.env.add(
                swift.Slider(
                    cb=lambda value, i=i: self.joint_slider_callback(value, i),
                    min=q_min_deg,
                    max=q_max_deg,
                    step=1,
                    value=q_initial_deg,
                    desc=f"Joint {i + 1}",
                    unit="°"
                )
            )

        self.env.add(swift.Button(cb=self.open_gripper, desc="OPEN GRIPPER"))
        self.env.add(swift.Button(cb=self.close_gripper, desc="CLOSE GRIPPER"))
        self.env.add(swift.Button(cb=self.print_pose, desc="PRINT CURRENT POSE"))
        self.env.add(swift.Button(cb=self.save_pose, desc="SAVE POSE"))
        self.env.add(swift.Button(cb=self.reset_pose, desc="RESET TO START"))

        self.update_labels()

    def run(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        self.cell.add_to_env(self.env)
        self.station.add_to_env(self.env)
        self.robot.add_to_env(self.env)

        self.update_visuals()

        self.env.set_camera_pose(
            [1.6, 1.8, 2.0],
            [0.0, -0.5, 0.8]
        )

        self.add_gui()
        self.env.step(0)

        print("\nFAIRINO FR3 Swift Teach Mode")
        print("-" * 72)
        print("Move Joint 1-Joint 6 with the Swift sliders.")
        print("Use OPEN/CLOSE GRIPPER for the gripper state.")
        print("Press SAVE POSE at each useful waypoint.")
        print("The tray is parked at the same FR3 stop as the normal task.")
        print("\nSuggested sequence:")
        print(
            "ready -> pre-pick -> pick -> lift -> carry-clear -> "
            "above-tray -> place -> clear-tray -> ready"
        )
        print("\nCtrl+C in the terminal to finish.")

        try:
            while True:
                self.update_visuals()
                self.update_labels()
                self.env.step(0.05)

        except KeyboardInterrupt:
            print("\nTeach mode finished.")

            if self.saved_poses:
                print("\nSaved pose summary:")
                for entry in self.saved_poses:
                    print(
                        f"Pose {entry['number']}: "
                        f"{np.round(np.degrees(entry['q']), 2).tolist()} "
                        f"| gripper="
                        f"{'closed' if entry['gripper_closed'] else 'open'}"
                    )
            else:
                print("No poses were saved.")


if __name__ == "__main__":
    FR3SwiftTeach().run()
