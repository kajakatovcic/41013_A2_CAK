# Teach/jog interface for the Smooth Oper-Caterer workcell 
#
# Adds a TEACH/JOG section to the Swift side panel and reads a gamepad with
# pygame:
#
#   MODES
#     AUTO   - the task sequence runs (TaskSequencer). Manual inputs are ignored.
#     MANUAL - "AUTO/MANUAL" button. The task sequence and conveyor are
#              paused and the chosen robot follows the operator. The
#              "Slider mode" select chooses what the six sliders do:
#                * Joint jog: one slider per joint (degrees). The joint
#                  moves smoothly towards the slider at a capped speed.
#                * Cartesian XYZ: the first three sliders are the TCP's world
#                  X, Y and Z (m). The TCP moves in a straight line towards the
#                  slider target at a capped speed with its orientation held,
#                  using the same RMRC step as the joystick. 
#                * Joystick Cartesian jogging (in either slider mode):
#                  left stick = X/Y, Triangle = Z up, Circle = Z down. The
#                  velocity is turned into joint motion with the Lab 9 RMRC
#                  step rather than Lab 11's fixed-damping step (see
#                  _cartesian_q for explanation)
#              Switching back to AUTO re-plans every move in progress from where
#              the operator left the arms so automatic motion never jumps.
#
#   PHYSICAL E-STOP: the joystick's CROSS (X) button.
#     It is active in BOTH modes all the time. Pressing it latches an e-stop
#     source "JOYSTICK X" in the safety controller. Releasing the button only 
#     releases that source. The cell stays stopped until RESET and START.
#     If the joystick is unplugged while connected it is also treated as an
#     e-stop (fail-safe). On Xbox this button is A. 
#   RUMBLE: while a manual jog is pushing a joint against its limit, the
#     joystick rumbles and the panel will specify the joint. It will also
#     rumble when a jog step is blocked by a predicted collision.
#
#   SAFETY OF MANUAL MOTION:
#     Every jot step is is checked with the shared planner's check() before 
#     appplied (joint limits, counter clearance + collision check).
#     Jogging is only allowed when the cell is IDLE, RUNNING OR COMPLETE.
#   No busy waiting as the joystick is polled once per frame (pygame.event.get,
#   non-blocking) inside the main loop which env.step(DT) paces.

import os
import time

import numpy as np
from scipy import linalg

from motion_planner import DT, DLS_LAMBDA_MAX
from safety_controller import IDLE, RUNNING, COMPLETE

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

AUTO = "AUTO"
MANUAL = "MANUAL"
JOINT = "Joint jog"
CARTESIAN = "Cartesian XYZ"
SLIDER_MODES = (JOINT, CARTESIAN)


# Joystick mapping: PS4/DualShock layout as in Lab 11 
ESTOP_BUTTON = 0  # Cross (X) -> PHYSICAL E-STOP (always active)
Z_DOWN_BUTTON = 1 # Circle -> tool down
Z_UP_BUTTON = 3 # Triangle -> tool up
AXIS_X = 0 # left stick left/right -> world X
AXIS_Y = 1 # left stick up/down -> world Y (stick up = +Y)
DEADZONE = 0.1 # Lab 11 joystick_deadzone_threshold
ESTOP_SOURCE = "JOYSTICK X"
DISCONNECT_SOURCE = "JOYSTICK DISCONNECTED"

# Manual-motion limits. Manual jogging happens beside people so it is slower
# than the planned motion 
JOG_SPEED = 0.10 # m/s, tool speed for joystick Cartesian jogs
JOINT_JOG_SPEED = np.deg2rad(20) # rad/s, joint slider jog speed
LIMIT_MARGIN = np.deg2rad(0.5) # "at the limit" if this close to it
CART_RANGE = 0.30 # m, Cartesian slider range either side of the TCP
CART_STEP = 0.005 # m, Cartesian slider resolution
CART_ARRIVED = 0.001 # m, Cartesian slider target reached
CART_TRACK_TOL = 0.01 # m, TCP lagging its target by more = cannot follow
CART_SETTLE_FRAMES = int(1.0 / DT) # frames allowed for the TCP to catch up
RUMBLE_STRENGTH = 0.8 # 0-1, both motors
RUMBLE_MS = 200 # re-triggered every frame while the condition holds
RUMBLE_PULSE = 0.5 # s, minimum rumble for each new limit/blocked event


class Joystick:
    """
    pygame gamepad polled once per frame (Lab 11). Handles a controller
    being plugged in or out while running. If pygame or a controller is not
    available the rest of the interface still works Only the physical
    e-stop and joystick jogging are unavailable and is statedi n the panel
    """

    def __init__(self):
        self.pg = None
        self.joy = None
        self.status = "pygame not installed"
        self.rumbling = False
        self.rumble_ok = None # None = not tried yet, then True/False from pygame
        self._pulse_until = 0.0
        try:
            import pygame
            pygame.init()
            pygame.joystick.init()
            self.pg = pygame
            self.status = "no joystick connected"
            if pygame.joystick.get_count() > 0:
                self._connect(0)
        except ImportError:
            pass

    def _connect(self, index):
        self.joy = self.pg.joystick.Joystick(index)
        self.joy.init()
        self.status = f"connected: {self.joy.get_name()}"

    @property
    def connected(self):
        return self.joy is not None

    def poll(self):
        """
        Process pygame's queued events (non-blocking) and read the controller.
        Returns (axes, buttons, event) where event is None, "connected" or
        "disconnected". Axes have the Lab 11 deadzone applied and missing
        axes/buttons are padded with zeros (like lab 11)
        """
        event = None
        if self.pg is None:
            return [0.0] * 4, [0] * 4, None
        for e in self.pg.event.get():
            if e.type == self.pg.JOYDEVICEADDED and self.joy is None:
                self._connect(e.device_index)
                event = "connected"
            elif e.type == self.pg.JOYDEVICEREMOVED and self.joy is not None \
                    and e.instance_id == self.joy.get_instance_id():
                self.joy = None
                self.rumbling = False
                self.status = "joystick DISCONNECTED"
                event = "disconnected"
        if self.joy is None:
            return [0.0] * 4, [0] * 4, event
        axes = [self.joy.get_axis(i) for i in range(self.joy.get_numaxes())]
        axes = [0.0 if abs(a) < DEADZONE else a for a in axes]
        buttons = [self.joy.get_button(i) for i in range(self.joy.get_numbuttons())]
        axes += [0.0] * (4 - len(axes))
        buttons += [0] * (4 - len(buttons))
        return axes, buttons, event

    def pulse(self, seconds=RUMBLE_PULSE):
        """Rumble for at least 'seconds' from now even if the condition that
        caused it only lasted one frame (e.g. a slider jog reaching a limit)."""
        self._pulse_until = max(self._pulse_until, time.monotonic() + seconds)

    def rumble(self, on):
        """
        Called every frame. Rumble while 'on' is True or a pulse is running,
        otherwise stop. Records whether the controller/driver accepts rumble.
        Joystick.rumble returns False when it is not supported.
        """
        if self.joy is None:
            return
        if on or time.monotonic() < self._pulse_until:
            ok = bool(self.joy.rumble(RUMBLE_STRENGTH, RUMBLE_STRENGTH, RUMBLE_MS))
            if ok != self.rumble_ok:
                self.rumble_ok = ok
                if not ok:
                    print("[JOYSTICK] rumble NOT supported by this controller/driver "
                          "(pygame Joystick.rumble returned False).")
            self.rumbling = True
        elif self.rumbling:
            self.joy.stop_rumble()
            self.rumbling = False


class TeachPendant:
    """
    The teach/jog interface- Swift panel elements, the joystick, and the
    manual-motion logic for the selected robot.

    param sequencer: the TaskSequencer (its robot lanes are the jog-able robots)
    param safety: the SafetyController (e-stop input and whether jogging is allowed)
    """

    def __init__(self, sequencer, safety, joystick=None):
        self.sequencer = sequencer
        self.safety = safety
        self.lanes = sequencer.robot_lanes
        self.k = 0 # index of the selected robot lane
        self.mode = AUTO
        self.slider_mode = JOINT
        self.joystick = Joystick() if joystick is None else joystick
        self._estop_held = False
        self.joint_target = {} # joint index -> target angle (rad) from a slider
        self.cart_target = None # world TCP target (m) from the Cartesian sliders
        self._settle = 0 # frames the TCP has been catching up to a reached target
        self.warning = ""
        self._axes, self._buttons = [0.0] * 4, [0] * 4
        self._hold = None # Cartesian jog target: (lane, TCP position, orientation)
        self._next_hold = None
        self._near_singular = False
        self._was_limited = False
        self._frame = 0
        self._labels = {}
        self.sliders = []
        self._shown = {}

    # Properties used by the run loop
    @property
    def auto_active(self):
        """True when the task sequence should run (AUTO mode)."""
        return self.mode == AUTO

    @property
    def lane(self):
        return self.lanes[self.k]

    @property
    def jog_allowed(self):
        # Not between RESET and START. A reset alone must not allow motion.
        return self.safety.state in (IDLE, RUNNING, COMPLETE)

    # Swift panel
    def add_to_panel(self, env):
        import swift

        def label(key, text):
            self._labels[key] = swift.Label(text)
            env.add(self._labels[key])

        label("header", "TEACH/JOG")
        env.add(swift.Button(cb=lambda _=None: self.toggle_mode(),
                             desc="AUTO/MANUAL"))
        label("mode", "")
        env.add(swift.Select(cb=self._on_select_robot, desc="Robot",
                             options=[l.name for l in self.lanes], value=0))
        env.add(swift.Select(cb=self._on_select_slider_mode, desc="Slider mode",
                             options=list(SLIDER_MODES), value=0))
        label("pose", "")
        # Six sliders, reused by both slider modes: rebuilt in range, label
        # and value whenever the robot or the slider mode changes.
        n = max(l.robot.n for l in self.lanes)
        for j in range(n):
            s = swift.Slider(cb=lambda v, j=j: self._on_slider(j, v),
                             min=-180, max=180, step=1, value=0,
                             desc=f"J{j + 1}", unit="deg")
            self.sliders.append(s)
            env.add(s)
        label("joystick", "")
        label("mapping", "Joystick: left stick = X/Y, Triangle = Z up, Circle = Z down, "
                         "CROSS (X) = PHYSICAL E-STOP (always active)")
        label("warning", "")
        env.add(swift.Button(cb=lambda _=None: self.test_rumble(), desc="Test joystick rumble"))
        self._sync_sliders(force=True)
        self._update_labels(force=True)

    def _set_label(self, key, text):
        """Only send a label to Swift when its text actually changes."""
        if key in self._labels and self._shown.get(key) != text:
            self._labels[key].desc = text
            self._shown[key] = text

    # Panel callbacks (run inside env.step on the main thread)
    def toggle_mode(self):
        if self.mode == AUTO:
            self.mode = MANUAL
            self.safety.note("MANUAL robot control: task sequence and conveyor paused.")
        else:
            self.mode = AUTO
            self._clear_jog()
            self.joystick.rumble(False)
            # The operator may have moved the arms. Re-plan the moves in
            # progress from where they are now.
            self.sequencer.replan_current()
            self.safety.note("AUTO: task sequence continues from the current arm poses.")
        self._update_labels(force=True)

    def test_rumble(self):
        """Panel button: rumble for one second to check the controller."""
        if not self.joystick.connected:
            self._warn("No joystick connected - cannot test rumble.")
            return
        self.joystick.pulse(1.0)
        self._warn("Rumble test: the joystick should rumble for 1sec.")

    def _on_select_robot(self, value):
        self.k = int(float(value))
        self._clear_jog()
        self.joystick.rumble(False)
        self._sync_sliders(force=True)
        self._update_labels(force=True)

    def _on_select_slider_mode(self, value):
        self.slider_mode = SLIDER_MODES[int(float(value))]
        self._clear_jog()
        self._sync_sliders(force=True)
        self._update_labels(force=True)

    def _on_slider(self, j, value):
        n_used = 3 if self.slider_mode == CARTESIAN else self.lane.robot.n
        if self.mode != MANUAL or j >= n_used:
            self._sync_sliders(force=True) # snap the slider back
            if self.mode != MANUAL:
                self._warn("Press AUTO/MANUAL to switch to MANUAL before jogging.")
            elif self.slider_mode == CARTESIAN:
                self._warn("Only the X, Y and Z sliders are used in Cartesian mode.")
            return
        if self.slider_mode == CARTESIAN:
            # Start from the current target (another axis may be moving) or the TCP
            if self.cart_target is None:
                self.cart_target = self.lane.planner.tcp(self.lane.robot.q).t.copy()
            self.cart_target[j] = float(value)
            return
        lo, hi = self.lane.robot.qlim[0][j], self.lane.robot.qlim[1][j]
        target = np.deg2rad(float(value))
        if target < lo or target > hi:
            self._warn(f"J{j + 1} target outside its limits: clamped to the limit.")
        self.joint_target[j] = float(np.clip(target, lo, hi))

    # Per-frame work called from the run loop
    def read_inputs(self):
        """
        Poll the joystick. Called EVERY frame in both modes BEFORE the
        safety controller ticks, so a physical e-stop is acted on in the same
        frame it is pressed.
        """
        axes, buttons, event = self.joystick.poll()
        if event == "disconnected":
            self.safety.press_estop(DISCONNECT_SOURCE)
            if self._estop_held:
                self.safety.release_estop(ESTOP_SOURCE)
                self._estop_held = False
        elif event == "connected" and DISCONNECT_SOURCE in self.safety.estop_sources:
            self.safety.release_estop(DISCONNECT_SOURCE)

        pressed = bool(buttons[ESTOP_BUTTON]) if self.joystick.connected else False
        if pressed and not self._estop_held:
            self.safety.press_estop(ESTOP_SOURCE)
        elif not pressed and self._estop_held:
            self.safety.release_estop(ESTOP_SOURCE)
        self._estop_held = pressed
        self._axes, self._buttons = axes, buttons

    def step(self):
        """One frame of manual motion for the selected robot (MANUAL mode only)."""
        self._frame += 1
        if self.mode != MANUAL or not self.jog_allowed:
            if self.joint_target:
                self._clear_jog() # discard pending jogs on any stop
            self._was_limited = False
            self.joystick.rumble(False) # still plays a running pulse (e.g. rumble test)
            self._update_labels()
            return

        lane = self.lane
        q = np.array(lane.robot.q, dtype=float)
        at_limit = blocked = False

        # 1. joint slider targets: move each towards its target at a capped speed
        if self.joint_target:
            q_new = q.copy()
            for j, target in list(self.joint_target.items()):
                dq = np.clip(target - q[j], -JOINT_JOG_SPEED * DT, JOINT_JOG_SPEED * DT)
                q_new[j] = q[j] + dq
                if abs(target - q_new[j]) < 1e-4:
                    del self.joint_target[j]
            at_limit, blocked = self._apply(lane, q, q_new, "joint jog")
            if blocked:
                self.joint_target.clear()
            self._hold = None # a Cartesian jog after this starts from here

        # 2. Cartesian slider target: straight line towards it at JOG_SPEED
        elif self.cart_target is not None:
            p_now = self._hold[1] if self._hold is not None and self._hold[0] is lane \
                else lane.planner.tcp(q).t
            d = self.cart_target - p_now
            dist = float(np.linalg.norm(d))
            # Arrived = the ACTUAL TCP is at the target. Near a singularity DLS
            # makes the TCP lag its moving target, so once the target reaches
            # the slider value the step keeps correcting (v = 0) for a while.
            short = float(np.linalg.norm(self.cart_target - lane.planner.tcp(q).t))
            self._settle = self._settle + 1 if dist < CART_ARRIVED else 0
            if short < CART_ARRIVED or self._settle > CART_SETTLE_FRAMES:
                if short >= CART_ARRIVED:
                    self._warn(f"{lane.name}: stopped {short * 1000:.0f} mm short of the X/Y/Z "
                               "target (near a singularity).")
                self.cart_target = None
                self._hold = None
            else:
                v = d / dist * min(JOG_SPEED, dist / DT) if dist >= CART_ARRIVED else np.zeros(3)
                q_new = self._cartesian_q(lane, q, v)
                at_limit, blocked = self._apply(lane, q, q_new, "Cartesian slider jog")
                if blocked or at_limit:
                    self._clear_jog() # stop and take the next target from here
                else:
                    self._hold = self._next_hold
                    lag = np.linalg.norm(lane.planner.tcp(lane.robot.q).t - self._hold[1])
                    if lag > CART_TRACK_TOL:
                        # e.g. target out of reach: the TCP stops following
                        self._warn(f"{lane.name}: TCP cannot follow the X/Y/Z target "
                                   "(out of reach or near a singularity) - jog stopped.")
                        self._clear_jog()
                        at_limit = True # rumble like a limit
                    elif self._near_singular:
                        self._warn(f"{lane.name}: near a singularity - Cartesian jog slowed "
                                   "(damped least squares). Joint-jog away from it.")

        # 3. Cartesian jog with the joystick (Lab 11 q3 mapping)
        else:
            v = np.zeros(3)
            if self.joystick.connected:
                ax, bt = self._axes, self._buttons
                v = JOG_SPEED * np.array([ax[AXIS_X], -ax[AXIS_Y],
                                          bt[Z_UP_BUTTON] - bt[Z_DOWN_BUTTON]])
                v = v / max(1.0, np.linalg.norm(v) / JOG_SPEED) # cap the combined speed
            if v.any():
                q_new = self._cartesian_q(lane, q, v)
                at_limit, blocked = self._apply(lane, q, q_new, "Cartesian jog")
                if blocked or at_limit:
                    self._hold = None # re-take the target from where the arm stopped
                else:
                    self._hold = self._next_hold
                    if self._near_singular:
                        self._warn(f"{lane.name}: near a singularity - Cartesian jog slowed "
                                   "(damped least squares). Joint-jog away from it.")
            else:
                self._hold = None # not jogging: next jog starts from the actual pose

        if (at_limit or blocked) and not self._was_limited:
            self.joystick.pulse() # every new limit/blocked event is felt
        self._was_limited = at_limit or blocked
        self.joystick.rumble(at_limit or blocked)
        self._sync_sliders()
        self._update_labels()

    def _cartesian_q(self, lane, q, v):
        """
        One frame of Cartesian jogging at tool velocity v (world frame).

        Lab 11 Q3 maps the joystick to a velocity and integrates
        q += dq*dt with a FIXED damping (lambda = 0.1), open loop. 
        This is tuned for the puma560 that overdamps on these much smaller arms.
        So the jog uses RMRC step instead (same loop as motion_planner.rmrc).
        i.e. a target pose is advanced by v*dt each frame, the commanded 
        velocity is (target - actual)/dt so errors are corrected + the orientation
        is held at its value when the jog started (a held cup stays upright) and DLS
        is only used near a singularity.
        """
        robot, s = lane.robot, lane.planner.settings
        if self._hold is None or self._hold[0] is not lane:
            T = lane.planner.tcp(q)
            self._hold = (lane, T.t.copy(), T.R.copy())
        _, p_target, Rd = self._hold
        p_target = p_target + v * DT
        flange_target = p_target - Rd @ s.tool.t # the tool is a pure translation
        T = robot.fkine(q).A
        linear_velocity = (flange_target - T[:3, 3]) / DT
        Ra = T[:3, :3]
        S = ((Rd - Ra) / DT) @ Ra.T # skew symmetric (Lab 9)
        angular_velocity = np.array([S[2, 1], S[0, 2], S[1, 0]])
        xdot = np.hstack((linear_velocity, angular_velocity))
        J = robot.jacob0(q)
        m = np.sqrt(abs(linalg.det(J @ J.T)))
        m_lambda = (1 - m / s.manip_epsilon) * DLS_LAMBDA_MAX if m < s.manip_epsilon else 0
        # Near a singularity the damping deliberately slows the arm instead of
        # spinning joints fast. the operator is told so a slow jog is expected.
        self._near_singular = m < s.manip_epsilon
        inv_j = linalg.inv(J.T @ J + m_lambda * np.eye(robot.n)) @ J.T
        self._next_hold = (lane, p_target, Rd)
        return q + (inv_j @ xdot) * DT

    def _apply(self, lane, q, q_new, what):
        """
        Check a jog step before applying it. Returns (at_limit, blocked).
        A step that would pass a joint limit stops at the limit (and rumbles).
        A step the planner's check rejects (counter, collision) is not applied.
        """
        robot = lane.robot
        lo, hi = robot.qlim[0], robot.qlim[1]
        over = np.where((q_new < lo + LIMIT_MARGIN) & (q_new < q) |
                        (q_new > hi - LIMIT_MARGIN) & (q_new > q))[0]
        if over.size:
            names = ", ".join(f"J{j + 1}" for j in over)
            self._warn(f"{lane.name}: {names} at joint limit - jog stopped (rumble).")
            q_new = np.clip(q_new, lo, hi)
            if np.allclose(q_new, q):
                return True, False
        reason = lane.planner.check(np.array([q, q_new]), lane.holding)
        if reason is not None:
            self._warn(f"{lane.name}: {what} blocked - {reason}.")
            return bool(over.size), True
        lane.robot.q = q_new
        if not over.size and self.warning:
            self.warning = ""
        return bool(over.size), False

    # Display helpers
    def _warn(self, text):
        self.warning = text
        self._set_label("warning", f"Jog: {text}")

    def _clear_jog(self):
        self.joint_target.clear()
        self.cart_target = None
        self._settle = 0
        self._hold = None

    def _sync_sliders(self, force=False):
        """Keep the sliders showing the selected robot's joints (joint mode)
        or TCP position (Cartesian mode), unless the operator is jogging with
        them. Joint sliders use the robot's joint limits. Cartesian sliders
        span CART_RANGE either side of the TCP and re-centre if it leaves."""
        jogging = self.joint_target or self.cart_target is not None
        if not self.sliders or (jogging and not force):
            return
        if not force and self._frame % 5:
            return
        robot = self.lane.robot
        if self.slider_mode == CARTESIAN:
            tcp = self.lane.planner.tcp(robot.q).t
            for j, s in enumerate(self.sliders):
                if j < 3:
                    if force or not s.min <= tcp[j] <= s.max:
                        s.min, s.max = round(tcp[j] - CART_RANGE, 3), round(tcp[j] + CART_RANGE, 3)
                        s.step, s.unit = CART_STEP, "m"
                        s.desc = f"{self.lane.name} TCP {'XYZ'[j]} (world)"
                    if force or abs(s.value - tcp[j]) > CART_STEP:
                        s.value = round(float(tcp[j]), 3)
                elif force:
                    s.min, s.max, s.step, s.value, s.unit = 0, 0, 1, 0, ""
                    s.desc = "- (not used in Cartesian mode)"
            return
        q = np.rad2deg(robot.q)
        for j, s in enumerate(self.sliders):
            if j < robot.n:
                lo, hi = np.rad2deg(robot.qlim[0][j]), np.rad2deg(robot.qlim[1][j])
                if force:
                    s.min, s.max = round(lo), round(hi)
                    s.step, s.unit = 1, "deg"
                    s.desc = f"{self.lane.name} J{j + 1}"
                if force or abs(s.value - q[j]) > 0.5:
                    s.value = round(float(q[j]), 1)

    def _update_labels(self, force=False):
        if not self._labels or (not force and self._frame % 4):
            return
        lane = self.lane
        if self.mode == AUTO:
            mode = "MODE: AUTO (task sequence). Press AUTO/MANUAL to jog."
        elif self.jog_allowed:
            mode = (f"MODE: MANUAL - jogging {lane.name}, sliders = {self.slider_mode} "
                    "(sequence and conveyor paused)")
        else:
            mode = (f"MODE: MANUAL - jog BLOCKED ({self.safety.state}): "
                    "RESET then START before jogging")
        self._set_label("mode", mode)
        tcp = lane.planner.tcp(lane.robot.q).t
        q = np.round(np.rad2deg(lane.robot.q), 1)
        self._set_label("pose", f"{lane.name}: q = {list(q)} deg | TCP = "
                                f"({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}) m")
        js = self.joystick.status
        if self.joystick.connected:
            js += " | physical e-stop " + ("PRESSED" if self._estop_held else "armed")
            if self.joystick.rumble_ok is False:
                js += " | rumble NOT supported by this controller/driver"
            elif self.joystick.rumble_ok:
                js += " | rumble ok"
        else:
            js += " | physical e-stop UNAVAILABLE"
        self._set_label("joystick", f"Joystick: {js}")
        if not self.warning:
            self._set_label("warning", "Jog: ok")
