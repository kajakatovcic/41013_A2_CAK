"""
FAIRINO FR3 Candidate Pose Test
-------
Quickly test candidate FR3 joint configurations similar to my previous lab assignment.

IMPORTANT:
- Passing the checks in this script does NOT certify a pose as physically safe.
- Checks joint limits and an approximate link-frame floor clearance.
- Does not model self-collision, tooling, payload, or external obstacles yet.
- Uses the complete mesh/collision model later for final workcell validation.

Edit/add PRESET_POSES or use console entry mode to try your own positions.

NOTE: I HAD SO MANY ISSUES WITH GIT SO I WAS WORKING ON AN INDIVIDUAL LOCAL FILE SO MY GITCOMMITS ARE NOT CONSISTENT 
THROUGH THE WEEKS BUT I SWEAR I HAVE BEEN WORKING ON THIS ASSIGNMENT :,(
"""

import numpy as np
import swift

from math import pi
from roboticstoolbox import jtraj
from ir_support import CylindricalDHRobotPlot

from fairino_fr3_model import create_fairino_fr3


class FairinoFR3PoseTest:
    def __init__(self):
        print("FAIRINO FR3 Candidate Pose Test")

        self.env = None
        self.robot = None

        # Candidate poses only.
        # These are intended to help visually explore the model.
        self.PRESET_POSES = {
            "ready": np.deg2rad([0, -90, 90, -90, -90, 0]),
            "compact": np.deg2rad([0, -90, 120, -120, -90, 0]),
            "forward": np.deg2rad([0, -60, 90, -120, -90, 0]),
            "raised": np.deg2rad([0, -80, 100, -110, -90, 0]),
        }

    # ------------------------------------------------------------------------- #
    def setup_scene(self):
        self.env = swift.Swift()
        self.env.launch(realtime=True)

        base_robot = create_fairino_fr3()

        cyl_viz = CylindricalDHRobotPlot(
            base_robot,
            cylinder_radius=0.035,
            color="blue",
        )
        self.robot = cyl_viz.create_cylinders()

        self.env.add(self.robot)
        self.env.set_camera_pose([1.4, 1.4, 1.1], [0, 0, 0.25])

        # Start in a bent candidate pose rather than q = 0.
        self.robot.q = self.PRESET_POSES["ready"].copy()

        self.env.step()

    # ------------------------------------------------------------------------- #
    def within_joint_limits(self, q):
        """
        Check q against robot.qlim.

        robot.qlim is 2 x n:
            row 0 = lower limits
            row 1 = upper limits
        """
        lower = self.robot.qlim[0, :]
        upper = self.robot.qlim[1, :]

        return np.all(q >= lower) and np.all(q <= upper)

    # ------------------------------------------------------------------------- #
    def joint_limit_report(self, q):
        lower = self.robot.qlim[0, :]
        upper = self.robot.qlim[1, :]

        print("\nJoint-limit report:")
        for i in range(self.robot.n):
            q_deg = np.degrees(q[i])
            lo_deg = np.degrees(lower[i])
            hi_deg = np.degrees(upper[i])

            state = "OK" if lo_deg <= q_deg <= hi_deg else "OUTSIDE LIMIT"

            print(
                f"J{i+1}: {q_deg:8.2f} deg   "
                f"[{lo_deg:8.2f}, {hi_deg:8.2f}]   {state}"
            )

    # ------------------------------------------------------------------------- #
    def link_frame_positions(self, q):
        """
        Get the origin of every DH frame using fkine_all(q).

        These are frame origins, not the complete physical link geometry.
        """
        transforms = self.robot.fkine_all(q)

        positions = []

        for T in transforms:
            positions.append(np.array(T.A[:3, 3], dtype=float))

        return np.asarray(positions)

    # ------------------------------------------------------------------------- #
    def approximate_floor_check(self, q, floor_z=0.0, margin=0.02):
        """
        Very simple screening check:
        warn if a non-base DH frame falls close to/below the floor.

        This is NOT a replacement for collision geometry.
        """
        positions = self.link_frame_positions(q)

        # Ignore frame 0/base origin because it is expected to be on the floor/base.
        moving_frame_positions = positions[1:]

        minimum_z = np.min(moving_frame_positions[:, 2])

        print(f"\nLowest non-base DH frame z = {minimum_z:.4f} m")

        if minimum_z < floor_z:
            print("WARNING: a DH frame is below the floor plane.")
            return False

        if minimum_z < floor_z + margin:
            print(
                "CAUTION: a DH frame is very close to the floor. "
                "Inspect the cylinder/mesh geometry carefully."
            )
            return False

        print("Approximate DH-frame floor-clearance check passed.")
        return True

    # ------------------------------------------------------------------------- #
    def print_pose_information(self, q):
        print("\nJoint values [rad]:")
        print(np.round(q, 6))

        print("\nJoint values [deg]:")
        print(np.round(np.degrees(q), 3))

        print("\nForward kinematics:")
        print(self.robot.fkine(q))

        self.joint_limit_report(q)
        self.approximate_floor_check(q)

    # ------------------------------------------------------------------------- #
    def check_whole_trajectory(self, q_trajectory):
        """
        Check EVERY sampled configuration, not only the final pose.
        """
        min_z = float("inf")

        for step, q in enumerate(q_trajectory):
            if not self.within_joint_limits(q):
                print(f"\nTrajectory rejected: joint limit exceeded at sample {step}.")
                return False

            positions = self.link_frame_positions(q)
            sample_min_z = np.min(positions[1:, 2])
            min_z = min(min_z, sample_min_z)

            if sample_min_z < 0:
                print(
                    f"\nTrajectory warning: a DH frame drops below z=0 "
                    f"at sample {step}."
                )
                return False

        print(
            f"\nWhole sampled trajectory passed the basic checks. "
            f"Minimum non-base DH-frame z = {min_z:.4f} m"
        )
        return True

    # ------------------------------------------------------------------------- #
    def move_to_test_pose(self, q_test, steps=80):
        q_test = np.asarray(q_test, dtype=float)

        if q_test.shape != (6,):
            raise ValueError("FR3 test pose must contain exactly 6 joint values.")

        if not self.within_joint_limits(q_test):
            print("\nTarget pose is outside the FR3 joint limits.")
            self.joint_limit_report(q_test)
            return False

        q_start = self.robot.q.copy()
        q_trajectory = jtraj(q_start, q_test, steps).q

        print("\nChecking the complete interpolated movement before animation...")

        if not self.check_whole_trajectory(q_trajectory):
            print(
                "\nThe basic screening check found a problem. "
                "The motion will not be animated."
            )
            return False

        for q in q_trajectory:
            self.robot.q = q
            self.env.step(0.04)

        self.print_pose_information(self.robot.q)

        print(
            "\nVisual check still required: inspect every link, not only the end effector."
        )

        return True

    # ------------------------------------------------------------------------- #
    def enter_pose_degrees(self):
        print(
            "\nEnter six joint values in degrees separated by spaces.\n"
            "Example: 0 -90 90 -90 -90 0"
        )

        raw = input("q1 q2 q3 q4 q5 q6 = ").strip()

        values = [float(v) for v in raw.replace(",", " ").split()]

        if len(values) != 6:
            raise ValueError("Please enter exactly six joint angles.")

        return np.deg2rad(values)

    # ------------------------------------------------------------------------- #
    def run(self):
        self.setup_scene()

        self.print_pose_information(self.robot.q)

        while True:
            print("\n----------------------------------------------")
            print("Choose a candidate pose to test:")
            for key in self.PRESET_POSES:
                print(f"  {key}")
            print("  custom")
            print("  quit")

            choice = input("\nSelection: ").strip().lower()

            if choice == "quit":
                break

            if choice == "custom":
                try:
                    q_test = self.enter_pose_degrees()
                except ValueError as exc:
                    print(exc)
                    continue

            elif choice in self.PRESET_POSES:
                q_test = self.PRESET_POSES[choice].copy()

            else:
                print("Unknown selection.")
                continue

            print("\nCandidate target [deg]:")
            print(np.round(np.degrees(q_test), 2))

            input(
                "\nPress Enter to check and animate the complete movement "
                "to this candidate pose.\n"
            )

            self.move_to_test_pose(q_test, steps=80)

        input("\nPose testing finished. Press Enter to exit.\n")


if __name__ == "__main__":
    FairinoFR3PoseTest().run()
