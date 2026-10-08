# Motion planning for the JAKA MiniCobo coffee station.
#
# The JAKA uses TWO trajectory methods based on the task:
#   jtraj (quintic, joint space) via plan_jtraj()
#       This is used for FREE-SPACE transits with no cup, or an empty cup. e.g.
#       home -> ready, ready -> pre-pick, cup dispenser -> coffee machine, return.
#       IK is solved only at the two ends meaning the motion is smooth with zero velocity
#       and acceleration at both ends. No Jacobian is inverted so passing near a singularity
#       causes no problem. The main limitation is that the tool path between the ends is 
#       not controlled.
#
#   RMRC (resolved motion rate control, Cartesian) via plan_rmrc()
#       This is used where the tool path and orientation are critical e.g. 
#       picking up/placing a full cup
#       RMRC tracks a Cartesian straight line or arc AND a fixed orientation at
#       every step which cannot be achieved with jtraj/quintic trajectories.
#
#   Measured on this robot:
#       * jtraj between the same two poses sags 6-12mm BELOW the straight line.
#         The cup slides into the bay 5 mm above the drip grate so jtraj would
#         drive it into the grate. RMRC stays within 0.1mm of the line.
#       * jtraj caps JOINT speed, not tool speed. On the full-cup carry its
#         peak TCP speed reached 0.28 m/s, above the 0.25m/s reduced-speed
#         figure. RMRC sets the TCP speed directly (0.15m/s cap).
#       * Both kept the cup within 0.1 deg of upright on these moves because
#         joints 2, 3 and 5 pitch in one plane. RMRC enforces it every step
#         rather than relying on that.
#
# Both of these are pure planning methods. They return a joint matrix and never
# touch robot.q or env.step(). The task sequence (in coffee_demo.py) animates them
# one frame at a time so an e-stop can stop motion between any two frames.
#
# Collision checking (collisions.py) combines two lab methods:
#   * Lab 5 line-plane intersection on the centre-line of every link, the
#     gripper and the held cup. 
#   * Lab 6 ellipsoids around the gripper and the held cup, tested against
#     points on the obstacle surfaces. These give the tool its real width and
#     height, which is important when entering tight spaces.
#   The arm links use the line test only, because they stay well clear of the
#   props and ellipsoids on all seven links would multiply the checking cost.
# When a planned move would collide the planner ACTIVELY AVOIDS rather than
# only stopping.
#   * jtraj: inserts a raised Cartesian via-point solved with ikine_LM
#   * RMRC: adds a vertical sine "hop" to the Cartesian path

from math import pi, ceil

import numpy as np
from scipy import linalg
from spatialmath import SE3
from spatialmath.base import rpy2r
from roboticstoolbox import jtraj, trapezoidal

from collisions import first_collision

# Timing and safe-speed choices
DT = 0.05 # control period/swift step (s)

# jtraj: fine interpolation step limit. jtraj() function is called repeatedly until
# no joint moves more than this per step. 1.5deg per 0.05s = 30deg/sec peak joint
# speed. This is a conservative speed for a cobot arm demo
MAX_JOINT_STEP = np.deg2rad(1.5)

# RMRC: tool-centre-point (TCP) speed cap. 0.15m/s carrying hot coffee is 
# below the generally used 0.25m/s reduced-speed figure for cobot operation.
# Trapezoidal profile peaks at 1.5x the average. accounted for below.
TCP_SPEED = 0.15 # m/s
YAW_SPEED = np.deg2rad(40)  # rad/s (how fast the cup may swing round)

# Gripper and grasp geometry
# A parallel two finger gripper with TCP 0.11m out along the flange z axis
# at the centre of the closed fingers
GRIPPER_LENGTH = 0.11
TOOL = SE3(0, 0, GRIPPER_LENGTH)

# Side grasped cup geometry. The cup is held from the side with the tool z-axis
# horizontal (pointing at cup) and tool x-axis pointing down. As long as tool x
# stays vertical the cup will stay upright regardless of yaw.
# MiniCobo joint 5's +-120 deg limit caps the tool-down envelope at ~0.30m above 
# the base
SIDE_PITCH = pi / 2

# RMRC settings
# When used in lab 9 the orientation rows were scaled by 0.1 because only the 
# path mattered. Here, orientation is critical (upright cup) so all six rows
# are weighted equally.
RMRC_W = np.diag([1, 1, 1, 1, 1, 1])
# Manipulability threshold for switching to DLS. 
# Lab 9 uses 0.1 for puma560. The MiniCobo is smaller so the threshold is scaled
# down
MANIP_EPSILON = 0.004
DLS_LAMBDA_MAX = 0.05 # Lab 9: (1 - m/epsilon)*0.05

# if joint limits clamp the solution so the cup ends off target/tilted, the plan
# is rejected.
RMRC_POS_TOL = 0.01 # m
RMRC_TILT_TOL = np.deg2rad(3.0) # rad

# Search steps the planner takes to find a collision-free path
VIA_LIFTS = (0.08, 0.14, 0.20) # jtraj via-point lift above the straight line (m)
HOP_HEIGHTS = (0.06, 0.10, 0.14) # RMRC sine hop amplitudes (m)

# Collision ellipsoids (Lab 6 Q2) for the parts a centre-line cannot represent.
# Radii are along the TOOL axes: x points down (cup axis), y is the finger
# closing direction, z points forward from the flange.
#
# Why ellipsoids here and not line-plane: the gripper is 0.10 m wide and the
# cup 0.09 m wide. Their centre-lines can pass 4-5cm from a wall that the real
# part would hit, so a line test alone either misses those contacts. 
# An ellipsoid carries the width.
# Why not ellipsoids alone: the obstacle side is a point cloud, and a sampled
# surface can be passed between points. 
#
# Gripper (coffee_station.Gripper): palm 0.07x0.10x0.04 m, fingers reach
# 0.125m from the flange, closed fingers 0.05m either side of the axis. The
# ellipsoid is centred halfway along (0.0625m) and sized so the closed
# fingertip corners (0.015, 0.05, 0.0625 from the centre) are inside:
# (0.015/0.05)^2 + (0.05/0.08)^2 + (0.0625/0.09)^2 = 0.96 < 1.
# It is still narrower than the 0.20m bay (0.08 < 0.10 each side).
GRIPPER_ELLIPSOID_OFFSET = 0.0625 # along tool z from the flange (m)
GRIPPER_ELLIPSOID_RADII = (0.05, 0.08, 0.09) # tool x, y, z (m)
# Cup  centred at its mid-height. Sized so the rim edge is inside:
# (0.0575/0.085)^2 + (0.046/0.065)^2 = 0.96 < 1. The ellipsoid hangs a little
# below the cup's base. the surfaces the cup rests on are deliberately not
# obstacles, so that does not read as a collision.
CUP_ELLIPSOID_RADII = (0.085, 0.065, 0.065) # tool x (vertical), y, z (m)


def side_grasp_pose(base, r, phi, h):
    """
    World TCP pose for a side grasp at polar coords (r, phi, h) in the robot
    BASE frame. 
    r= hoirzontal distance from base axis
    phi = bearing from base x axis
    h = height above base plate
    Tool points radially outward so the cup is alwasy approached head on

    Defining every station target this way keeps them all in one coordinate
    frame (the base) and they move with the robot if the base pose changes
    """
    return base * SE3(r * np.cos(phi), r * np.sin(phi), h) * SE3.Rz(phi) * SE3.Ry(SIDE_PITCH)


def cup_tilt(T_tcp):
    """Angle (rad) between the held cup's axis and world vertical.
    The cup axis is the tool's -x axis (see SIDE GRASP above)"""
    R = T_tcp.A[:3, :3] if hasattr(T_tcp, "A") else np.asarray(T_tcp)[:3, :3]
    return float(np.arccos(np.clip(-R[2, 0], -1.0, 1.0)))


class JakaPlanner:
    """
    jtraj and RMRC planning for the JAKA MiniCobo with joint-limit, floor and
    collision checks + active avoidance.

    param robot: the JakaMiniCobo (its .base is the station frame)
    param obstacles: list of collisions.Obstacle shared with the scene. The
        list is read at plan time so obstacles added or moved later are seen
    param min_link_z: world height every link frame (except the base) must
        stay above. Set to the counter top plus a margin for the link radius
    """

    def __init__(self, robot, obstacles, min_link_z):
        self.robot = robot
        self.obstacles = obstacles
        self.min_link_z = min_link_z
        # Holding state changes the tool chain used for collision checks.
        self.cup_height = 0.115
        self.grasp_height = 0.055

    # Shared helpers
    @property
    def base(self):
        return self.robot.base

    def tcp(self, q):
        """Forward kinematics to the TCP"""
        return self.robot.fkine(q) * TOOL

    def polar_of(self, T_tcp):
        """(r, phi, h) of a world TCP pose in the base frame"""
        p = (self.base.inv() * T_tcp).t
        return float(np.hypot(p[0], p[1])), float(np.arctan2(p[1], p[0])), float(p[2])

    def tool_frames(self, holding):
        """
        Frames appended after the flange for collision checking:
        flange -> TCP (the gripper), then (if a cup is held) a segment down the
        cup's axis from just above its base to its rim. These centre-lines go
        through the Lab 5 line-plane test as the exact backstop; their width
        is covered by tool_ellipsoids() below.
        """
        def frames(T_flange):
            T_tcp = T_flange @ TOOL.A
            out = [T_tcp]
            if holding:
                down = T_tcp[:3, 0] # tool +x points down the cup
                low = T_tcp.copy() # 1cm above the cup's base
                low[:3, 3] = T_tcp[:3, 3] + down * (self.grasp_height - 0.01)
                high = T_tcp.copy() # the cup's rim
                high[:3, 3] = T_tcp[:3, 3] - down * (self.cup_height - self.grasp_height)
                out += [low, high]
            return out
        return frames

    def tool_ellipsoids(self, holding):
        """
        Lab 6 collision ellipsoids for the gripper and (if held) the cup,
        as a function flange_T (4x4) -> [(T_ellipsoid, radii), ...].

        Each ellipsoid's frame is placed with forward kinematics plus a fixed
        offset (the way Lab 6 attaches an ellipsoid to each link). Its
        axes follow the tool axes so the radii in GRIPPER_ELLIPSOID_RADII and
        CUP_ELLIPSOID_RADII are along tool x (down), y (fingers), z (forward).
        """
        T_grip = SE3(0, 0, GRIPPER_ELLIPSOID_OFFSET).A
        # Cup centre: the cup's base is grasp_height below the TCP along tool x,
        # so its mid-height is (grasp_height - cup_height / 2) along tool x.
        T_cup = (TOOL * SE3(self.grasp_height - self.cup_height / 2, 0, 0)).A

        def ellipsoids(T_flange):
            out = [(T_flange @ T_grip, GRIPPER_ELLIPSOID_RADII)]
            if holding:
                out.append((T_flange @ T_cup, CUP_ELLIPSOID_RADII))
            return out
        return ellipsoids

    def within_joint_limits(self, q):
        """ikine_LM does not check qlim so every solution is checked."""
        lower, upper = self.robot.qlim[0], self.robot.qlim[1]
        return bool(np.all((np.asarray(q) >= lower) & (np.asarray(q) <= upper)))

    def path_clears_counter(self, q_matrix):
        """Every link frame except the base stays above min_link_z.
        Frame 0 is the base plate, which sits ON the counter,
        so it is skipped."""
        for q in q_matrix:
            frames = self.robot.fkine_all(q)
            if min(f.t[2] for f in list(frames)[1:]) < self.min_link_z:
                return False
        return True

    def check(self, q_matrix, holding):
        """
        Full validity check of a joint matrix. Returns None if it is safe,
        otherwise a short reason string (shown on the GUI fault display).
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
    
    # jtraj 
    @staticmethod
    def fine_jtraj(q1, q2, max_step=MAX_JOINT_STEP):
        """
        Lab 5 fine_interpolation - keep calling jtraj with more steps until
        no joint moves more than max_step per step. This turns the joint-speed
        limit into a step count, so every jtraj move obeys the safe speed.
        Starts from a sensible estimate instead of 2 to save iterations.
        """
        steps = max(2, int(np.max(np.abs(np.asarray(q2) - np.asarray(q1))) / max_step))
        while np.any(max_step < np.abs(np.diff(jtraj(q1, q2, steps).q, axis=0))):
            steps += 1
        return jtraj(q1, q2, steps).q

    def solve_ik(self, T_tcp, q_start, q_guesses=(), seed=0):
        """
        Solve inverse kinematics using ikine_LM from several seeds with a
        fixed random seed (repeatable), joint limits enforced, and the solution
        with the LEAST joint travel from q_start kept. This way the arm stays on its
        current branch. ikine_LM solves for the FLANGE so the TCP target is
        converted with the inverse tool transform.
        """
        T_flange = T_tcp * TOOL.inv()
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
        Plan a quintic joint-space move from q_start to goal.

        param goal: a joint vector (a named configuration or a supplied
                    joint state), or an SE3 TCP pose solved with ikine_LM.

        Returns (q_matrix, note): q_matrix is None if no safe plan exists and note
        explains what happened ("direct", "via +0.14 m", or the failure reason).
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
            return q_matrix, "direct"

        # ACTIVE AVOIDANCE (seen in lab5): route through a Cartesian via-point
        # above the midpoint of the straight line between the two TCP poses,
        # solved with ikine_LM. Higher lifts are tried until a clear route is
        # found. Both legs are checked.
        T_a, T_b = self.tcp(q_start), self.tcp(q_goal)
        r_a, phi_a, h_a = self.polar_of(T_a)
        r_b, phi_b, h_b = self.polar_of(T_b)
        dphi = np.arctan2(np.sin(phi_b - phi_a), np.cos(phi_b - phi_a))
        phi_m = phi_a + dphi / 2
        r_m = max(0.45, (r_a + r_b) / 2)
        for lift in VIA_LIFTS:
            T_via = side_grasp_pose(self.base, r_m, phi_m, max(h_a, h_b) + lift)
            q_via = self.solve_ik(T_via, q_start, [q_goal] + list(q_guesses))
            if q_via is None:
                continue
            leg1 = self.fine_jtraj(q_start, q_via)
            leg2 = self.fine_jtraj(q_via, q_goal)
            if self.check(leg1, holding) is None and self.check(leg2, holding) is None:
                return np.vstack([leg1, leg2[1:]]), f"re-planned via +{lift:.2f} m ({reason})"
        return None, f"no safe route ({reason})"

    # RMRC 
    def cartesian_path(self, T_start, goal_polar, mode, hop=0.0):
        """
        Build the Cartesian TCP path (3 x steps) and yaw profile for an RMRC move
        using a trapezoidal scalar s.

        mode 'line': straight line in the world (approach, insert, lower, retreat)
        mode 'arc' : polar interpolation about the base axis. Radius, bearing and
                     height change together so the cup swings round the robot at
                     a controlled radius instead of cutting through the base.
        hop: amplitude of an added vertical bump h += hop*sin(pi*s) (lab 9 path
             shape) used to lift over an obstacle in the way.
        Steps are chosen from the path length and yaw change so the peak TCP
        speed stays under TCP_SPEED. A trapezoidal profile peaks at 1.5x its
        average speed. 1.6 is used to leave margin for the discrete steps.
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
                    ceil(1.6 * length / (TCP_SPEED * DT)),
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
        RMRC applied to the TCP path x (3 x steps) with orientation roll = 0, 
        pitch = SIDE_PITCH, yaw = yaw[i].

        Uses lab 9 loop with two differences:
          * the TCP path is first converted to a FLANGE path (because fkine and
            jacob0 describe the flange). With a fixed tool offset the flange point
            is the TCP point minus GRIPPER_LENGTH along the tool z-axis.
          * the loop starts from the robot's ACTUAL joint state q0 rather than
            an ikine_LM solution so a re-plan after a stop continues smoothly
            from wherever the arm halted.

        Returns (q_matrix, info) where info holds the worst manipulability and
        the final position and tilt errors (for the GUI and the report)
        """
        steps = x.shape[1]
        Rd_all = [rpy2r(0, SIDE_PITCH, yaw[i]) for i in range(steps)]
        xf = np.zeros([3, steps])
        for i in range(steps):
            xf[:, i] = x[:, i] - GRIPPER_LENGTH * Rd_all[i][:, 2]

        q_matrix = np.zeros([steps, self.robot.n])
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
            if m[i] < MANIP_EPSILON: # damped least squares near
                m_lambda = (1 - m[i] / MANIP_EPSILON) * DLS_LAMBDA_MAX # a singularity 
            else:
                m_lambda = 0
            inv_j = linalg.inv(J.T @ J + m_lambda * np.eye(self.robot.n)) @ J.T
            qdot = inv_j @ xdot
            for j in range(self.robot.n): # stop a joint that
                if q_matrix[i, j] + delta_t * qdot[j] < qlim[j, 0]: # would pass its limit
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
            "max_tilt": max(cup_tilt(self.tcp(q)) for q in q_matrix[:: max(1, steps // 20)]),
        }
        return q_matrix, info

    def plan_rmrc(self, q_start, goal_polar, mode="line", holding=False):
        """
        Plan an RMRC move from the arm's current joint state to a TCP goal given
        in base-frame polar coordinates (r, phi, h), side-grasp orientation.

        If the straight/arc path would collide, re-plan with a vertical sine hop
        of increasing height (active avoidance). Returns (q_matrix, note).
        """
        q_start = np.asarray(q_start, dtype=float)
        T_start = self.tcp(q_start)
        last_reason = None
        for hop in (0.0,) + HOP_HEIGHTS:
            x, yaw = self.cartesian_path(T_start, goal_polar, mode, hop)
            q_matrix, info = self.rmrc(q_start, x, yaw)
            if info["pos_err"] > RMRC_POS_TOL or info["max_tilt"] > RMRC_TILT_TOL:
                last_reason = (f"RMRC could not track the path "
                               f"(err {info['pos_err'] * 1000:.0f} mm, "
                               f"tilt {np.rad2deg(info['max_tilt']):.1f} deg)")
                continue
            reason = self.check(q_matrix, holding)
            if reason is None:
                note = "direct" if hop == 0 else f"re-planned with {hop:.2f} m hop ({last_reason})"
                return q_matrix, note
            last_reason = reason
        return None, f"no safe path ({last_reason})"
