# Shared Swift set-up and run loop for every station demo and main scene
#
# Swift panel controls (the simulated GUI e-stop and safety inputs):
#   E-STOP/Release E-stop/RESET/START-RESUME
#   Hand into light curtain - asynchronous unsafe-zone signal at the exit
#   plus any station-specific buttons (e.g. the JAKA's milk pitcher)

from safety_controller import SafetyController, LightCurtain
from motion_planner import DT
from workcell_layout import LIGHT_CURTAIN_X, EXIT_HALF_WIDTH, LIGHT_CURTAIN_Z


def make_curtain():
    """The light curtain's beams at the exit (see safety_controller.LightCurtain)."""
    return LightCurtain(LIGHT_CURTAIN_X, EXIT_HALF_WIDTH - 0.06, LIGHT_CURTAIN_Z)


def launch(cell, parts, camera):
    """
    Start Swift and add the shared workcell + each robot/station. 
    camera = (position, look_at).
    """
    import swift
    env = swift.Swift()
    env.launch(realtime=True)
    cell.add_to_env(env)
    for part in parts:
        part.add_to_env(env)
    env.set_camera_pose(*camera)
    return env


def make_safety(cell):
    """
    The safety controller which is connected to the stack light, the curtain field and
    a state label in the Swift panel. 
    Returns (safety, state_label, task_label).
    """
    import swift
    state_label = swift.Label("STATE: IDLE")
    task_label = swift.Label("Task: waiting for START")
    safety = SafetyController(beacon=cell.beacon, curtain_field=cell.curtain_field,
                              label=state_label)
    return safety, state_label, task_label


def obstacle_button(desc, toggle, sequencer, safety, what):
    """
    A panel button that moves a station's movable obstacle into/out of a
    robot's path (the JAKA's milk pitcher). It tells the sequencer that 
    the world changed so the current move is rechecked for collisions and 
    replanned around the obstacle.
    toggle = the stations 'toggle' method and returns True when now in the
    path.
    Returns a (description, callback) pair for add_controls
    """
    def cb():
        placed = toggle()
        sequencer.notify_world_changed()
        safety.note(f"{what} placed in the path." if placed else f"{what} removed.")
    return desc, cb


def add_controls(env, safety, cell, state_label, task_label, extra_buttons=(), pendant=None):
    """The safety panel. extra_buttons: (description, callback) pairs.
    pendant: optional teach_pendant.TeachPendant whose TEACH/JOG section is
    added below the safety controls."""
    import swift

    def toggle_hand(_=None):
        inside = cell.toggle_hand()
        safety.note("Hand placed in the light curtain." if inside else "Hand removed from the light curtain.")

    env.add(state_label)
    env.add(task_label)
    env.add(swift.Button(cb=lambda _=None: safety.press_estop("GUI"), desc="E-STOP"))
    env.add(swift.Button(cb=lambda _=None: safety.release_estop(), desc="Release E-stop"))
    env.add(swift.Button(cb=lambda _=None: safety.reset(), desc="RESET"))
    env.add(swift.Button(cb=lambda _=None: safety.start(), desc="START / RESUME"))
    env.add(swift.Button(cb=toggle_hand, desc="Hand into light curtain (toggle)"))
    for desc, cb in extra_buttons:
        env.add(swift.Button(cb=lambda _=None, cb=cb: cb(), desc=desc))
    if pendant is not None:
        pendant.add_to_panel(env)


def run(env, safety, cell, curtain, sequencer, task_label, pendant=None):
    """
    Main running loop - One iteration per frame as env.step(DT) paces the loop
    (it sleeps), so nothing here spins the processor while the cell is stopped.

    With a teach pendant each frame: Read the joystick (the physical e-stop is
    queued BEFORE the safety controller ticks so it acts in the same frame),
    tick the safety controller, do one frame of manual jogging, and run the
    task sequence only in AUTO mode.
    """
    sequencer.update_visuals()
    env.step(0)
    print("Swift running. Use the panel buttons. Ctrl+C to exit.")
    last_status = None
    try:
        while True:
            if pendant is not None:
                pendant.read_inputs()
            safety.tick()
            hand = [cell.hand_obstacle] if cell.hand_inside else []
            safety.update_curtain(curtain.is_broken(hand))
            if pendant is not None:
                pendant.step()
            if pendant is None or pendant.auto_active:
                sequencer.tick()
            sequencer.update_visuals()
            if sequencer.status != last_status:
                last_status = sequencer.status
                task_label.desc = f"Task: {sequencer.status}"
            env.step(DT)
    except KeyboardInterrupt:
        print("Exiting.")
