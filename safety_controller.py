# Cell safety state machine (e-stop, light curtain, faults, reset and resume)
#
#            START                      E-STOP (any state)
#   IDLE ----------> RUNNING ---------------------------------> ESTOP
#                    |  ^   \  light curtain broken               |
#                    |  |    -------------------> PROTECTIVE_STOP |
#                    |  |   unavoidable collision -> FAULT        |
#                    |  |                               |         |
#                    |  +--------- START ---- RESET_OK <-- RESET -+
#                    |                                  (only when the e-stop is
#                    +--> COMPLETE                       released AND the curtain
#                                                        is clear)
#
# This file implements the state machine and hte display of the safety state. 
#   * Prompt halt: the sequencer only advances a trajectory while the state is
#     RUNNING so the arm and conveyor stop on the very next frame.
#   * Visible state: stack-light beacon colour, light-curtain field colour, a
#     state label in the Swift sim + terminal messages
#   * Releasing the e-stop does NOT restart anything and does not clear a fault.
#     the operator must RESET to acknowledge the stop and then START to resume
#   * RESET (acknowledge) and START (resume) are separate.
#   * On START the sequencer re-plans the interrupted move from the joint state
#     where the harm actually stopped so there is no jump, and RMRC still follows
#     its straight line/upright constraint (e.g for cups)
#   * No busy wait as inputs are queued and processed once per frame in tick().
#     The main loop is paced by env.step(DT) which sleeps between frames
#
# Inputs can come from the Swift buttons (main thread) or a separate thread,
# e.g. the PLC listener for the physical e-stop on the real robot,
# because everything goes through a thread-safe queue.

import queue

import numpy as np

from collisions import segment_hit

IDLE = "IDLE"
RUNNING = "RUNNING"
ESTOP = "E-STOP"
PROTECTIVE_STOP = "PROTECTIVE STOP"
FAULT = "FAULT"
RESET_OK = "RESET - READY TO RESUME"
COMPLETE = "COMPLETE"

BEACON_COLOURS = {
    RUNNING: "green", # running
    IDLE: "amber", # waiting for the operator
    RESET_OK: "amber",
    PROTECTIVE_STOP: "amber",
    ESTOP: "red", # emergency stop
    FAULT: "red", # faulted
    COMPLETE: "blue", # cycle complete
}


class LightCurtain:
    """
    Simulated safety light curtain across the conveyor exit.
    Polled every frame (independent of the task). Behaves as an asynchronous input.
    On a real cell muting sensors let a tray pass the curtain without a trip. Here
    only intruders are tested and not trays - models that behaviour

    METHOD - Lab 6 Q1 ray casting with the Lab 5 line-plane test.
    A real light curtain is a row of infrared beams from an emitter post to a
    receiver post, and it trips when any beam is interrupted. That is modelled
    directly: each beam is a line segment between the posts, and it is tested
    against the faces of each intruder's RectangularPrism with
    line_plane_intersection and the inside-triangle check (collisions.segment_hit).
    Lab 6 Q1 casts sensor rays against a plane in the same way.

    Why line-plane and not ellipsoids here: a beam IS a line with no thickness,
    so the line test is the physically correct model, and it is exact. An
    ellipsoid test would need the intruder as a point cloud and the beam as a
    volume, approximating both. The beam spacing plays the role of the real
    curtain's resolution: an object smaller than the spacing can pass between
    beams, exactly as on real hardware (30 mm is a common hand-detection rating).
    """

    def __init__(self, x, y_half, z_range, beam_spacing=0.03, y_centre=0.0):
        z0, z1 = z_range
        self.beams = [(np.array([x, y_centre - y_half, z]), np.array([x, y_centre + y_half, z]))
                      for z in np.arange(z0 + beam_spacing / 2, z1, beam_spacing)]

    def broken_beam(self, intruders):
        """Height of the first interrupted beam, or None if every beam is clear.
        intruders: list of collisions.Obstacle (e.g. the attendee's hand)."""
        for obs in intruders:
            for start, end in self.beams:
                if segment_hit(start, end, obs) is not None:
                    return float(start[2])
        return None

    def is_broken(self, intruders):
        return self.broken_beam(intruders) is not None


class SafetyController:
    def __init__(self, beacon=None, curtain_field=None, label=None, log=print):
        self.state = IDLE
        self.message = "Press START to begin."
        self.estop_pressed = False
        self.curtain_broken = False
        self.resumed = False # set on START after a stop; read by the sequencer
        self._events = queue.Queue()
        self._beacon = beacon
        self._field = curtain_field
        self._label = label
        self._log = log
        self._show()

    # inputs (thread safe) called from the GUI or a separate thread (e.g. PLC listener)
    def press_estop(self, source="GUI"):
        self._events.put(("estop", source))

    def release_estop(self):
        self._events.put(("release", None))

    def reset(self):
        self._events.put(("reset", None))

    def start(self):
        self._events.put(("start", None))

    # inputs called from the main loop 
    def update_curtain(self, broken):
        """Poll result of the light curtain once per frame"""
        if broken != self.curtain_broken:
            self.curtain_broken = broken
            if self._field is not None:
                self._field.set("red" if broken else "green")
            if broken:
                if self.state in (RUNNING, RESET_OK):
                    self._go(PROTECTIVE_STOP, "Light curtain broken at the conveyor exit. "
                                              "Clear it, then RESET and START.")
                else:
                    self._note("Light curtain broken.")
            else:
                self._note("Light curtain clear.")

    def fault(self, message):
        """Raised by the planner/sequencer e.g. an unavoidable collision is detected"""
        if self.state != ESTOP:
            self._go(FAULT, message + " Remove the cause, then RESET and START.")

    def complete(self):
        if self.state == RUNNING:
            self._go(COMPLETE, "Order delivered to the collection zone")

    def note(self, message):
        """Informational message (e.g. 're-planned around milk pitcher')."""
        self._note(message)

    @property
    def motion_allowed(self):
        return self.state == RUNNING

    # Per frame tick() called from the main loop. THis is where the queued events are 
    # processed and the state machine is implemented. The main loop is paced by 
    # env.step(DT) which sleeps between frames.
    def tick(self):
        while True:
            try:
                kind, arg = self._events.get_nowait()
            except queue.Empty:
                break
            self._handle(kind, arg)

    def _handle(self, kind, arg):
        if kind == "estop":
            self.estop_pressed = True
            self._go(ESTOP, f"E-STOP pressed ({arg}). Release it, then RESET, then START.")
        elif kind == "release":
            if self.estop_pressed:
                self.estop_pressed = False
                self._note("E-stop released. Still stopped: press RESET to acknowledge.")
        elif kind == "reset":
            if self.state not in (ESTOP, FAULT, PROTECTIVE_STOP):
                self._note(f"RESET ignored in state {self.state}.")
            elif self.estop_pressed:
                self._note("Cannot RESET: e-stop is still pressed.")
            elif self.curtain_broken:
                self._note("Cannot RESET: light curtain is still blocked.")
            else:
                self._go(RESET_OK, "Reset acknowledged. Press START to resume.")
        elif kind == "start":
            if self.state == IDLE:
                self._go(RUNNING, "Running.")
            elif self.state == RESET_OK:
                self.resumed = True
                self._go(RUNNING, "Resumed: re-planning the interrupted move from the halted pose.")
            else:
                self._note(f"START ignored in state {self.state}"
                           + (": press RESET first." if self.state in (ESTOP, FAULT, PROTECTIVE_STOP) else "."))

    # Display
    def _go(self, state, message):
        self.state = state
        self.message = message
        self._log(f"[SAFETY] {state}: {message}")
        self._show()

    def _note(self, message):
        self.message = message
        self._log(f"[SAFETY] ({self.state}) {message}")
        self._show()

    def _show(self):
        if self._beacon is not None:
            self._beacon.set(BEACON_COLOURS[self.state])
        if self._label is not None:
            self._label.desc = f"STATE: {self.state} | {self.message}"
