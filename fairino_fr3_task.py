import numpy as np
from spatialmath import SE3

from fairino_fr3_motion import TOOL, GRASP_ROLL, GRASP_PITCH
from fairino_fr3_station import POSES, FR3Gripper

from task_sequencer import (
    Segment,
    RobotLane,
    ConveyorLane,
    TaskSequencer
)

from workcell_layout import COLLECTION_X
from workcell_scene import Tray


# -----------------------------------------------------------------------------#
# FAIRINO FR3 job
# -----------------------------------------------------------------------------#
READY_Q = np.deg2rad([
    0,
    -90,
    90,
    -90,
    -90,
    0
])

PICK_APPROACH_Q = np.deg2rad([
    104.2,
    -38.5,
    78.8,
    -40.2,
    104.2,
    -90.0
])

class FR3Job:

    def __init__(self, station, tray):

        self.station = station
        self.tray = tray
        self.drink = station.drink

        # Relative pose of the drink in the TCP frame while being held.
        # This gets calculated when the gripper finishes closing.
        self.drink_in_tcp = None

        # ---------------------------------------------------------
        # Conveyor stop position
        # ---------------------------------------------------------
        # Stop the tray so its DRINK_SLOT is aligned with the FR3
        # "place" target.
        r, phi, _ = POSES["place"]

        place_world = (
            station.base
            * SE3(
                r * np.cos(phi),
                r * np.sin(phi),
                0
            )
        )

        slot_x = place_world.t[0]

        self.tray_stop_x = (
            slot_x
            - Tray.DRINK_SLOT.t[0]
        )

    # -------------------------------------------------------------------------
    # Motion / action sequence
    # -------------------------------------------------------------------------

    def segments(self):

        S = Segment

        return [

            S(
                "jtraj",
                "home -> ready",
                goal=READY_Q
            ),

            S(
                "jtraj",
                "ready -> above cold drink",
                goal="above_pick"
            ),

            S(
                "rmrc",
                "lower gripper around cold drink",
                goal="pick",
                mode="line"
            ),

            S(
                "grip",
                "close gripper",
                close=True,
                hold=True
            ),

            S(
                "rmrc",
                "lift cold drink",
                goal="above_pick",
                mode="line"
            ),

            S(
                "rmrc",
                "carry drink level to tray",
                goal="above_tray",
                mode="arc",
                requires="tray_at_fr3"
            ),

            S(
                "rmrc",
                "lower drink onto tray",
                goal="place",
                mode="line"
            ),

            S(
                "grip",
                "release drink",
                close=False,
                hold=False
            ),

            S(
                "rmrc",
                "lift gripper clear of drink",
                goal="clear_tray",
                mode="line",
                sets="drink_placed"
            ),

            S(
                "jtraj",
                "return to ready",
                goal=READY_Q
            ),
        ]

    # -------------------------------------------------------------------------
    # Target information used by task_sequencer.py
    # -------------------------------------------------------------------------

    def target_pose(self, name):

        r, phi, h = POSES[name]

        return (
            self.station.base
            * SE3(
                r * np.cos(phi),
                r * np.sin(phi),
                h
            )
            * SE3.Rz(phi)
            * SE3.Ry(GRASP_PITCH)
            * SE3.Rx(GRASP_ROLL)
        )

    def target_polar(self, name):

        return POSES[name]

    def ik_seeds(self, name):

        return self.station.ik_seeds(name)

    # -------------------------------------------------------------------------
    # Non-motion actions
    # -------------------------------------------------------------------------

    def run_action(self, seg, lane):

        if seg.kind == "grip":

            gripper = self.station.gripper

            target = (
                FR3Gripper.CLOSED
                if seg.p["close"]
                else FR3Gripper.OPEN
            )

            step = (
                FR3Gripper.OPEN
                - FR3Gripper.CLOSED
            ) / 10

            # Move fingers gradually
            if abs(gripper.opening - target) <= step:
                gripper.opening = target
            else:
                gripper.opening += (
                    np.sign(target - gripper.opening)
                    * step
                )

            finished = gripper.opening == target

            # -----------------------------------------------------
            # Gripper has just closed
            # -----------------------------------------------------
            if finished and seg.p["close"]:

                T_flange = lane.robot.fkine(
                    lane.robot.q
                )

                T_tcp = T_flange * TOOL

                # Store the drink's exact relative pose
                # so it follows the TCP without jumping.
                self.drink_in_tcp = (
                    T_tcp.inv()
                    * self.drink.pose
                )

            # -----------------------------------------------------
            # Gripper has just opened
            # -----------------------------------------------------
            if finished and not seg.p["close"]:

                # Hand the drink to the tray at its exact
                # current pose.
                self.tray.attach(
                    self.drink,
                    self.tray.pose.inv()
                    * self.drink.pose
                )

                self.drink_in_tcp = None

            return finished

        return True

    # -------------------------------------------------------------------------
    # Update gripper + held drink every frame
    # -------------------------------------------------------------------------

    def update_visuals(self, lane):

        T_flange = lane.robot.fkine(
            lane.robot.q
        )

        self.station.gripper.update(
            T_flange
        )

        if (
            lane.holding
            and self.drink_in_tcp is not None
        ):

            T_tcp = T_flange * TOOL

            self.drink.held_by(
                T_tcp,
                self.drink_in_tcp
            )


# -----------------------------------------------------------------------------#
# Standalone FAIRINO FR3 task
# -----------------------------------------------------------------------------#

class FR3Task(TaskSequencer):

    def __init__(
        self,
        robot,
        planner,
        station,
        tray,
        safety,
        log=print
    ):

        self.tray = tray

        self.job = FR3Job(
            station,
            tray
        )

        self.tray_stop_x = (
            self.job.tray_stop_x
        )

        S = Segment

        conveyor = [

            S(
                "convey",
                "tray -> FAIRINO FR3 station",
                x=self.tray_stop_x,
                sets="tray_at_fr3"
            ),

            S(
                "convey",
                "tray -> collection zone",
                x=COLLECTION_X,
                requires="drink_placed",
                sets="delivered"
            ),
        ]

        lanes = [

            RobotLane(
                "FAIRINO FR3",
                robot,
                planner,
                self.job,
                self.job.segments()
            ),

            ConveyorLane(
                "conveyor",
                tray,
                conveyor
            ),
        ]

        super().__init__(
            safety,
            lanes,
            log
        )

    @property
    def holding(self):

        return self.lanes[0].holding

    def dry_run(self, lane=None):

        return super().dry_run(
            lane or self.lanes[0]
        )