"""
FAIRINO FR3 DRINK-HANDLING TASK
===============================

Defines the FR3 pick-and-place operation used in both:
    - the standalone FR3 demonstration
    - the complete multi-robot workcell

--------------
The sequence intentionally combines multiple robotics techniques:

    jtraj:
        free-space movement where only the start/end joint configurations
        need to be controlled.

    RMRC line:
        Cartesian approach, lift and placement where the TCP path needs to be
        straight and predictable.

    RMRC arc:
        carries the drink around the FR3 while keeping the payload upright.

    inverse kinematics:
        converts named Cartesian poses into reachable robot configurations.

    collision avoidance:
        all planned motion is checked before execution and can be re-planned
        if the simulated cooler is moved into the path.

The FR3 task also synchronises with the conveyor using TaskSequencer flags.
"""

import numpy as np
from spatialmath import SE3

from fairino_fr3_motion import TOOL, GRASP_ROLL, GRASP_PITCH
from fairino_fr3_station import POSES, FR3Gripper

from task_sequencer import Segment, RobotLane, ConveyorLane, TaskSequencer

from workcell_layout import COLLECTION_X
from workcell_scene import Tray

# -----------------------------------------------------------------------------#
# Working configuration 
# -----------------------------------------------------------------------------#

# Safe working configuration used before and after the drink operation.
READY_Q = np.deg2rad([0, -90, 90, -90, -90, 0])

# =============================================================================
# Integrated FR3 job
# =============================================================================

class FR3Job:
    """
    Robot-specific task interface used by the shared TaskSequencer.

    The job describes WHAT the FR3 should do.

    The shared planner/sequencer determines HOW each motion is planned and
    executed safely.
    """

    def __init__(self, station, tray):

        self.station = station
        self.tray = tray
        self.drink = station.drink

        # Exact relative drink pose captured when the fingers close.
        # Keeping this transform allows the drink to follow the true TCP
        # position without being forced into an artificial fixed pose.
        self.drink_in_tcp = None

        # ---------------------------------------------------------
        # Conveyor synchronisation
        # ---------------------------------------------------------
        # The stop position aligns the tray's defined DRINK_SLOT with the FR3
        # station along the conveyor X direction.

        self.tray_stop_x = (self.station.base.t[0] - Tray.DRINK_SLOT.t[0])

    def segments(self):

        """
        Define the ordered FR3 drink-handling operation.

        Each Segment also states which planning technique is required.
        """

        S = Segment

        return [
            # Free-space joint-space motion
            # Exact TCP path is not important here, therefore a smooth
            # quintic jtraj is appropriate.
            S("jtraj", "home -> ready", goal=READY_Q),

            # Another free-space motion toward the pickup region.
            # The target is supplied as a Cartesian pose and solved through
            # inverse kinematics before the joint trajectory is generated.
            S("jtraj", "ready -> above cold drink", goal="above_pick"),
            
            # Controlled Cartesian approach
            # RMRC is used because the end effector should descend in a
            # predictable straight line around the drink.
            S("rmrc", "lower gripper around cold drink", goal="pick", mode="line"),

            # Close the visual gripper and mark the lane as holding a payload.
            S("grip", "close gripper", close=True, hold=True),

            # Straight upward RMRC lift to clear the surrounding drink stock.
            S("rmrc", "lift cold drink", goal="above_pick", mode="line"),

            # Conveyor synchronisation + payload transport
            # The FR3 is allowed to pick the drink early, but it cannot begin
            # carrying toward the tray until the conveyor lane has reached
            # the FR3 station and set tray_at_fr3.
            
            # The arc mode moves around the base instead of cutting directly
            # through it and keeps the drink upright during transport.
            S("rmrc", "carry drink level to tray", goal="above_tray", mode="arc", requires="tray_at_fr3"),
            
            # Straight Cartesian lowering onto the designated tray position.
            S("rmrc", "lower drink onto tray", goal="place", mode="line"),
            
            # Open gripper and transfer drink ownership from robot to tray.
            S("grip", "release drink", close=False, hold=False),
            
            # Retreat vertically before allowing the conveyor to move.
            # drink_placed is deliberately set AFTER the gripper has cleared
            # the drink so downstream motion cannot start too early.
            S("rmrc", "lift gripper clear of drink", goal="clear_tray", mode="line", sets="drink_placed"),
            
            # Return to the safe working posture.
            S("jtraj", "return to ready", goal=READY_Q),
        ]

    # -------------------------------------------------------------------------
    # Cartesian Target information used by task_sequencer.py
    # -------------------------------------------------------------------------

    def target_pose(self, name):

        """
        Convert a named polar target into a full Cartesian TCP pose.

        x = r cos(phi)
        y = r sin(phi)

        The TCP yaw follows phi as the FR3 moves around its base.
        Roll and pitch remain fixed to preserve the validated top-down grasp.
        """

        r, phi, h = POSES[name]

        return (self.station.base * SE3( r * np.cos(phi), r * np.sin(phi), h) * SE3.Rz(phi) * SE3.Ry(GRASP_PITCH) * SE3.Rx(GRASP_ROLL))

    def target_polar(self, name):
        """
        RMRC uses the original polar target directly.
        """
        return POSES[name]

    def ik_seeds(self, name):
        """
        Obtain candidate initial configurations for inverse kinematics.
        """
        return self.station.ik_seeds(name)

    # -------------------------------------------------------------------------
    # Gripper action 
    # -------------------------------------------------------------------------

    def run_action(self, seg, lane):
        """
        Execute non-motion actions one simulation frame at a time.

        The fingers open/close gradually so the gripper operation is visible
        during the demonstration rather than changing instantaneously.
        """

        if seg.kind == "grip":

            gripper = self.station.gripper

            target = (
                FR3Gripper.CLOSED
                if seg.p["close"]
                else FR3Gripper.OPEN
            )

            # Ten visual increments between fully open and closed.
            step = (FR3Gripper.OPEN - FR3Gripper.CLOSED) / 10

            # Move fingers gradually
            if abs(gripper.opening - target) <= step:
                gripper.opening = target
            else:
                gripper.opening += (np.sign(target - gripper.opening) * step)

            finished = gripper.opening == target

            # -----------------------------------------------------
            # Gripper has just closed
            # -----------------------------------------------------
            if finished and seg.p["close"]:

                T_flange = lane.robot.fkine(lane.robot.q)
                T_tcp = T_flange * TOOL

                # Save the ACTUAL relative pose at the end of the grasp.
                # This prevents the drink from jumping to an assumed TCP pose.
                self.drink_in_tcp = (T_tcp.inv() * self.drink.pose)

            # -----------------------------------------------------
            # Gripper has just opened
            # -----------------------------------------------------
            if finished and not seg.p["close"]:

                # Convert the current world drink pose to a tray-relative pose.
                # Once attached, the drink travels with the conveyor tray.
                self.tray.attach(self.drink, self.tray.pose.inv() * self.drink.pose)
                self.drink_in_tcp = None

            return finished

        return True

    # -------------------------------------------------------------------------
    # Update gripper + held drink every frame
    # -------------------------------------------------------------------------

    def update_visuals(self, lane):
        """
        Keep the gripper and held drink synchronised with the current FR3 pose.
        """

        T_flange = lane.robot.fkine(lane.robot.q)
        self.station.gripper.update(T_flange)

        if (
            lane.holding
            and self.drink_in_tcp is not None
        ):

            T_tcp = T_flange * TOOL
            self.drink.held_by(T_tcp, self.drink_in_tcp)


# -----------------------------------------------------------------------------#
# Standalone FAIRINO FR3 task
# -----------------------------------------------------------------------------#

class FR3Task(TaskSequencer):
    """
    Standalone FR3 + conveyor task used by fairino_fr3_demo.py.

    The complete workcell does NOT create this independent sequencer.
    Instead main_scene.py inserts FR3Job into the shared sequencer alongside
    the other robots.

    This class exists so the FR3 subsystem can still be demonstrated and
    tested independently.
    """

    def __init__(self, robot, planner, station, tray, safety, log=print):

        self.tray = tray
        self.job = FR3Job(station, tray)
        self.tray_stop_x = (self.job.tray_stop_x)

        S = Segment

        conveyor = [
            # Standalone tray arrival releases the FR3 carry segment.
            S("convey", "tray -> FAIRINO FR3 station", x=self.tray_stop_x, sets="tray_at_fr3"),
            
            # Tray cannot leave until drink placement is safely complete.
            S("convey", "tray -> collection zone", x=COLLECTION_X, requires="drink_placed", sets="delivered"),
        ]

        lanes = [

            RobotLane("FAIRINO FR3", robot, planner, self.job, self.job.segments()),
            ConveyorLane("conveyor", tray, conveyor),
        ]

        super().__init__(safety, lanes, log)

    @property
    def holding(self):
        return self.lanes[0].holding

    def dry_run(self, lane=None):
        return super().dry_run(lane or self.lanes[0])