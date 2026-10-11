# Shared motion planner for every robot in the Smooth Oper-Caterer workcell
#
# Used by every arm. The planning methods are the same for every robot.
# Anything that differs is stored in a PlannerSettings object (tool, grasp orientation,
# speeds, collision shapes of tool and payload)
# Each robot keeps its own settings file, e.g. jaka_motion.py
#
# This planner uses two main trajectory methods
#
#   jtraj (quintic, joint space) via plan_jtraj()
#       Used for free-space transits. IK only at the two ends with smooth start and stop.
#       The tool path between the ends is not controlled, only used when nothing is close 
#       to te tool.
#
#   RMRC (resolved motion rate control, Cartesian) via plan_rmrc()
#       For moves where the tool path is important, e.g. straight approaches, lifts and
#       placements, and carrying a payload level at a capped tool speed.
#
# Both are pure planning methods (return a joint matrix and never touch
# robot.q or env.step()). task_sequencer.py animates them
# one frame at a time so an e-stop can stop motion between any two frames.
#
# collisions.py combines line-plane intersection on link/tool/payload centrelines
# with ellipsoids around the too land payload. When a planned move would collide
# the planner avoids. jtraj inserts a Cartesian waypoint.
# RMRC adds a vertical sine "hop" to the cartesian path
# Station targets are given in POLAR coordinates (r, phi, h) in the robot's
# base frame: 
# r = distance from the base axis
# phi = bearing from the base x-axis
# h = height above the base plate. 
# x = r cos(phi), y = r sin(phi) is circle around the robot calc from lab 4
# tool always faces along the bearing (yaw=phi) so one (r,phi,h) fully defines a pose

from math import pi, ceil

import numpy as np
from scipy import linalg
from spatialmath import SE3
from spatialmath.base import rpy2r
from roboticstoolbox import jtraj, trapezoidal

from collisions import first_collision

# Cell-wide timing and safe-speed choices (the same for every robot)
DT = 0.05 # control period/swift step (s)

# jtraj - fine interpolation step limit. jtraj() is called repeatedly until no
# joint moves more than this per step. 1.5deg per 0.05s = 30deg/s joint speed
# for safety
MAX_JOINT_STEP = np.deg2rad(1.5)

# RMRC - how fast a payload may swing round the robot
YAW_SPEED = np.deg2rad(40) # rad/s

# RMRC settings. 
# orientation is important so all 6 rows of W are equally weighted
RMRC_W = np.diag([1, 1, 1, 1, 1, 1])
DLS_LAMBDA_MAX = 0.05 # (1 - m/epsilon)*0.05

# If joint limits clamp the solution so the payload ends off target the plan
# is rejected
RMRC_POS_TOL = 0.01 # m
RMRC_TILT_TOL = np.deg2rad(3.0) # rad

# Search steps the planner takes to find a collision-free path
VIA_LIFTS = (0.08, 0.14, 0.20) # jtraj waypoint lift above the straight line (m)
HOP_HEIGHTS = (0.06, 0.10, 0.14) # RMRC sine hop amplitudes (m)


class PlannerSettings:
    """
    Everything regarding a robot's motion that differs between robots.

    param tool: SE3 from flange to TCP. This must be a pure translation. Leave
                as SE3() if the robot model already includes its tool in fkine.
                If used incorrectly the offset it applied twice
    param roll, pitch: fixed part of the tool orientation (following lab 9
                roll-pitch-yaw). Yaw follows the bearing phi. Orientation is
                R = Rz(yaw) * Ry(pitch) * Rx(roll) = rpy2r(roll, pitch, yaw).
                For a side grasp (tool horizontal, x down): r=0,p=pi/2
                For tool pointing down: r=pi,p=0
    param payload_up: direction in the TCP frame that points up through the 
                payload. used to measure tilt (for when an object its holding 
                needs to stay level)
    param tcp_speed: RMRC tool speed cap(m/s)
    param manip_epsilon: manipulatability below which DLS is used (from lab7/9).
                Scales with robot size so is set per robot.
    param via_min_radius: the closest a jtraj waypoint for avoidance may be to
                the base axis (inside it the robot cannot reach with this tool 
                orientation).
    param tool_chain/payload_chain: SE3 offsets from the flange. Added to link
                chain for line-plane test. payload_chain only used when holding
                something
    param tool_ellipsoids/payload_ellipsoids: (SE3 offset from flange, radii 
                along that frames axes) for ellispoid test (lab 6)
    """

    def __init__(self, tool=SE3(), roll=0.0, pitch=0.0, payload_up=(0, 0, 1),
                 tcp_speed=0.15, manip_epsilon=0.01, via_min_radius=0.30,
                 tool_chain=None, payload_chain=(), tool_ellipsoids=(),
                 payload_ellipsoids=()):
        self.tool = tool
        self.roll = roll
        self.pitch = pitch
        self.payload_up = np.asarray(payload_up, dtype=float)
        self.tcp_speed = tcp_speed
        self.manip_epsilon = manip_epsilon
        self.via_min_radius = via_min_radius
        if tool_chain is None:
            tool_chain = [tool] if np.linalg.norm(tool.t) > 0 else []
        self.tool_chain = list(tool_chain)
        self.payload_chain = list(payload_chain)
        self.tool_ellipsoids = list(tool_ellipsoids)
        self.payload_ellipsoids = list(payload_ellipsoids)


class MotionPlanner:
    """
    jtraj and RMRC planning with joint-limit, floor and collision checks +
    active avoidance. Used for all robots.
    
    param robot: robot model (.base is the station frame)
    param obstacles: list of collisions.Obstacle shared with the scene. List
        is used fo rplanning so obstacles added/moved later are seen
    param min_link_z: world height every link frame (except base) must stay
        above. set to counter top + margin for link radius
    param settings: robot's PlannerSettings
    """

    def __init__(self, robot, obstacles, min_link_z, settings):
        self.robot = robot
        self.obstacles = obstacles
        self.min_link_z = min_link_z
        self.settings = settings

    # Shared helpers
    @property
    def base(self):
        return self.robot.base

    def tcp(self, q):
        """Forward kinematics to the TCP"""
        return self.robot.fkine(q) * self.settings.tool

    def pose_at(self, r, phi, h):
        """
        World TCP pose at polar coords (r,phi,h) in base frame with the tool
        facing along hte bearing (yaw=phi) and this robot's fixed roll and 
        pitch. Defining every target this way keeps them in one frame (base frame)
        so they move with the robot if the base pose changes
        """
        s = self.settings
        return (self.base * SE3(r * np.cos(phi), r * np.sin(phi), h)
                * SE3.Rz(phi) * SE3.Ry(s.pitch) * SE3.Rx(s.roll))

    def polar_of(self, T_tcp):
        """(r, phi, h) of a world TCP pose in the base frame"""
        p = (self.base.inv() * T_tcp).t
        return float(np.hypot(p[0], p[1])), float(np.arctan2(p[1], p[0])), float(p[2])

    def payload_tilt(self, T_tcp):
        """Angle (rad) between the payload's up direction and world vertical"""
        R = T_tcp.A[:3, :3] if hasattr(T_tcp, "A") else np.asarray(T_tcp)[:3, :3]
        up = R @ self.settings.payload_up
        return float(np.arccos(np.clip(up[2] / np.linalg.norm(up), -1.0, 1.0)))

    def tool_frames(self, holding):
        """
        Frames added after flange for line-plane intersection test. Includes
        tool centre-line + payload centre-lines when holding. 
        """
        offsets = [T.A for T in self.settings.tool_chain]
        if holding:
            offsets += [T.A for T in self.settings.payload_chain]

        def frames(T_flange):
            return [T_flange @ T for T in offsets]
        return frames

    def tool_ellipsoids(self, holding):
        """
        Collision ellipsoids (from lab 6) for tool and payload as a function
        flange_T (4x4) -> [(T_ellipsoid, radii), ...].
        Each ellipsoid frame is the flange pose times a fixed offset
        """
        items = [(T.A, radii) for T, radii in self.settings.tool_ellipsoids]
        if holding:
            items += [(T.A, radii) for T, radii in self.settings.payload_ellipsoids]

        def ellipsoids(T_flange):
            return [(T_flange @ T, radii) for T, radii in items]
        return ellipsoids

    def within_joint_limits(self, q):
        """ikine_LM does not respect joint limits so every solution is checked"""
        lower, upper = self.robot.qlim[0], self.robot.qlim[1]
        return bool(np.all((np.asarray(q) >= lower) & (np.asarray(q) <= upper)))

    def path_clears_counter(self, q_matrix):
        """
        Every link frame except the base must stay above min_link_z.
        Frame 0 = base plate which is skipped as its on the counter
        """
        for q in q_matrix:
            frames = self.robot.fkine_all(q)
            if min(f.t[2] for f in list(frames)[1:]) < self.min_link_z:
                return False
        return True

    def check(self, q_matrix, holding):
        """
        Validity check of joint matrix. Returns None if safe or a short
        reason string (shown on GUI fualt display)
        """
        for q in (q_matrix[0], q_matrix[-1]):
            if not self.within_joint_limits(q):
                return "joint limit"
        if not self.path_clears_counter(q_matrix):
            return "counter (floor) clearance"
        hit = first_collision(self.robot, q_matrix, self.obstacles,
                              tool_frames=self.tool_frames(holding),
                              tool_ellipsoids=self.tool_ellipsoids(holding))
        if hit is not None:
            return f"collision with {hit}"
        return None

    # Quintic trajectory (jtraj)
    @staticmethod
    def fine_jtraj(q1, q2, max_step=MAX_JOINT_STEP):
        """
        fine_interpolation - keep calling jtraj wiht more steps until no joint
        moves more than max_step per step. Turns the joint-speed imit into a 
        step count so every jtraj move obeys the safe speed.
        Starts from a sensible estimate to save iterations
        """
        steps = max(2, int(np.max(np.abs(np.asarray(q2) - np.asarray(q1))) / max_step))
        while np.any(max_step < np.abs(np.diff(jtraj(q1, q2, steps).q, axis=0))):
            steps += 1
        return jtraj(q1, q2, steps).q

    def solve_ik(self, T_tcp, q_start, q_guesses=(), seed=0):
        """
        Solve IK with ikine_LM from several seeds - includes a fixed random
        seed for repeatability, enforced joint limits. The solution with the
        LEAST joint travel from q_start is kept so the arm stays on its 
        current branch. ikine_LM soves for flange only. TCP target is 
        converted with inverse tool transform
        """
        T_flange = T_tcp * self.settings.tool.inv()
        best = None
        for q0 in [np.asarray(q_start, dtype=float)] + [np.asarray(g, dtype=float) for g in q_guesses]:
            sol = self.robot.ikine_LM(T_flange, q0=q0, seed=seed)
            if not sol.success or not self.within_joint_limits(sol.q):
                continue
            travel = float(np.linalg.norm(sol.q - q_start))
            if best is None or travel < best[0]:
                best = (travel, sol.q)
        return None if best is None else best[1]

    def plan_jtraj(self, q_start, goal, holding=False, q_guesses=()):
        """
        Plan quintic joint space trajectory from q_start to goal.
        param goal: a joint vector (named config or supplied joint state
                    or an SE3 TCP pose solved with ikine_LM)
        Returns (q_matrix, note): q_matrix is None if no safe plan exists + a 
                    note of why ("direct", "via +0.14 m", or failure reason)
        """
        q_start = np.asarray(q_start, dtype=float)
        if isinstance(goal, SE3):
            q_goal = self.solve_ik(goal, q_start, q_guesses)
            if q_goal is None:
                return None, "IK failed (unreachable or out of joint limits)"
        else:
            q_goal = np.asarray(goal, dtype=float)
            if not self.within_joint_limits(q_goal):
                return None, "goal outside joint limits"

        q_matrix = self.fine_jtraj(q_start, q_goal)
        reason = self.check(q_matrix, holding)
        if reason is None:
            return q_matrix, "direct" # no waypoint needed

        # Active avoidance (from lab 5) - route through a cartesian waypoint
        # above the midpoint of the straight line between two TCP poses. Solved
        # with ikine_LM.
        # Tries higher lifts until a clear route is found. Check sboth legs
        T_a, T_b = self.tcp(q_start), self.tcp(q_goal)
        r_a, phi_a, h_a = self.polar_of(T_a)
        r_b, phi_b, h_b = self.polar_of(T_b)
        dphi = np.arctan2(np.sin(phi_b - phi_a), np.cos(phi_b - phi_a))
        phi_m = phi_a + dphi / 2
        r_m = max(self.settings.via_min_radius, (r_a + r_b) / 2)
        for lift in VIA_LIFTS:
            T_via = self.pose_at(r_m, phi_m, max(h_a, h_b) + lift)
            q_via = self.solve_ik(T_via, q_start, [q_goal] + list(q_guesses))
            if q_via is None:
                continue
            leg1 = self.fine_jtraj(q_start, q_via)
            leg2 = self.fine_jtraj(q_via, q_goal)
            if self.check(leg1, holding) is None and self.check(leg2, holding) is None:
                return np.vstack([leg1, leg2[1:]]), f"re-planned via +{lift:.2f}m ({reason})"
        return None, f"no safe route ({reason})"

    # RMRC
    def cartesian_path(self, T_start, goal_polar, mode, hop=0.0):
        """
        Build cartesian TCP path (3 x steps) and yaw profile for RMRC move.
        Uses trapezoidal scalar 's' (from labs 6 and 9)
        
        mode 'line': straight line (lifting, lowering etc)
        mode 'arc': polar interpolation about base axis. radius, bearing +
                    height change together so payload will swing around robot
                    at a controlled radius instead of cutting through the base
        hope: amplitude of added vertical bump h += hop*sin(pi*s) used to lift
            over an obstacle in the way
        Steps are chosen from path length and yaw change so that hte TCP speed
        stays under tcp_speed. Trapezoidal profile peaks at 1.5x average speed.
        1.6 is used to leave margin for discrete steps.
        """
        r0, phi0, h0 = self.polar_of(T_start)
        r1, phi1, h1 = goal_polar
        dphi = np.arctan2(np.sin(phi1 - phi0), np.cos(phi1 - phi0))

        p_start = T_start.t
        p_goal = (self.base * SE3(r1 * np.cos(phi1), r1 * np.sin(phi1), h1)).t

        def point(s):
            if mode == "line":
                p = (1 - s) * p_start + s * p_goal
            else:
                r = (1 - s) * r0 + s * r1
                phi = phi0 + s * dphi
                h = (1 - s) * h0 + s * h1
                p = (self.base * SE3(r * np.cos(phi), r * np.sin(phi), h)).t
            return p + np.array([0, 0, hop * np.sin(pi * s)])

        # Length estimate on a fine sampling of the path for the speed cap
        fine = np.array([point(s) for s in np.linspace(0, 1, 50)])
        length = float(np.sum(np.linalg.norm(np.diff(fine, axis=0), axis=1)))
        steps = max(10,
                    ceil(1.6 * length / (self.settings.tcp_speed * DT)),
                    ceil(1.6 * abs(dphi) / (YAW_SPEED * DT)))

        s = trapezoidal(0, 1, steps).q
        x = np.zeros([3, steps])
        yaw = np.zeros(steps)
        base_yaw = np.arctan2(self.base.R[1, 0], self.base.R[0, 0])
        for i in range(steps):
            x[:, i] = point(s[i])
            yaw[i] = base_yaw + phi0 + s[i] * dphi
        return x, yaw

    def rmrc(self, q0, x, yaw):
        """
        Using lab9 RMRC applied to TCP path x (3 x steps) with orientaiton
        rpy2r(roll, pitch, yaw[i]). Differences from lab 9:
            * TCP path is first converted to a flange path since fkine and
              jacob0 describe the flange. With fixed tool offset t (puure
              translation), the flange point is the TCP point minus R * t
            * loop starts from robot's joint state q0 rather than ikine_LM
              solution so a replan after a stop continues smoothly from
              wherever it originally stopped
        Returns (q_matrix, info). Info holds the worst manipulatability + the
        final position and tilt errors. Used in GUI and the report
        """
        s = self.settings
        steps = x.shape[1]
        Rd_all = [rpy2r(s.roll, s.pitch, yaw[i]) for i in range(steps)]
        xf = np.zeros([3, steps])
        for i in range(steps):
            xf[:, i] = x[:, i] - Rd_all[i] @ s.tool.t

        n = self.robot.n
        q_matrix = np.zeros([steps, n])
        q_matrix[0, :] = q0
        m = np.zeros(steps)
        qlim = np.transpose(self.robot.qlim)
        delta_t = DT

        for i in range(steps - 1):
            T = self.robot.fkine(q_matrix[i, :]).A # current flange pose
            delta_x = xf[:, i + 1] - T[:3, 3] # position error to next waypoint
            Rd = Rd_all[i + 1] # next desired orientation
            Ra = T[:3, :3] # current orientation
            Rdot = (1 / delta_t) * (Rd - Ra) # rotation matrix error
            S = Rdot @ Ra.T # skew symmetric
            linear_velocity = (1 / delta_t) * delta_x
            angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
            xdot = RMRC_W @ np.hstack((linear_velocity, angular_velocity))
            J = self.robot.jacob0(q_matrix[i, :])
            m[i] = np.sqrt(abs(linalg.det(J @ J.T))) # measure of manipulability
            if m[i] < s.manip_epsilon: # damped least squares near
                m_lambda = (1 - m[i] / s.manip_epsilon) * DLS_LAMBDA_MAX # a singularity
            else:
                m_lambda = 0
            inv_j = linalg.inv(J.T @ J + m_lambda * np.eye(n)) @ J.T
            qdot = inv_j @ xdot
            # stop a joint that would pass its limit: 
            for j in range(n):                                          
                if q_matrix[i, j] + delta_t * qdot[j] < qlim[j, 0]:     
                    qdot[j] = 0
                elif q_matrix[i, j] + delta_t * qdot[j] > qlim[j, 1]:
                    qdot[j] = 0
            q_matrix[i + 1, :] = q_matrix[i, :] + delta_t * qdot
        m[-1] = m[-2] if steps > 1 else 0

        T_end = self.tcp(q_matrix[-1])
        info = {
            "steps": steps,
            "min_manip": float(np.min(m)),
            "pos_err": float(np.linalg.norm(T_end.t - x[:, -1])),
            "max_tilt": max(self.payload_tilt(self.tcp(q)) for q in q_matrix[:: max(1, steps // 20)]),
        }
        return q_matrix, info

    def plan_rmrc(self, q_start, goal_polar, mode="line", holding=False):
        """
        Plan RMRC move from arm's current joint state to a TCP goal given in base
        frame polar coords (r,phi,h) in this robot's tool orientation.
        If the path would collide, replan with a sine hope of increasing height
        for active avoidance.
        Returns (q_matrix, note)
        """
        q_start = np.asarray(q_start, dtype=float)
        T_start = self.tcp(q_start)
        last_reason = None
        for hop in (0.0,) + HOP_HEIGHTS:
            x, yaw = self.cartesian_path(T_start, goal_polar, mode, hop)
            q_matrix, info = self.rmrc(q_start, x, yaw)
            if info["pos_err"] > RMRC_POS_TOL or info["max_tilt"] > RMRC_TILT_TOL:
                last_reason = (f"RMRC could not track the path "
                               f"(err {info['pos_err'] * 1000:.0f}mm, "
                               f"tilt {np.rad2deg(info['max_tilt']):.1f}deg)")
                continue
            reason = self.check(q_matrix, holding)
            if reason is None:
                note = "direct" if hop == 0 else f"re-planned with {hop:.2f}m hop ({last_reason})"
                return q_matrix, note
            last_reason = reason
        return None, f"no safe path ({last_reason})"
