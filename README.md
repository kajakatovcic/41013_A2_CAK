# Presenting: The Smooth Oper-Caterer - 41013 (Industrial Robotics)
SafeCo wants to present the safe, collaborative features of their robotic systems at conferences and meetings through an interactive catering system during lunch/tea breaks. Attendees can experience the complex capabilities of SafeCo’s systems as they are served their choice of food, beverages and accompaniments from the ‘Smooth Oper-Caterer’.

<div align="center">
  <a href="https://youtu.be/DWSJ-DbVl58?si=Tdz84tgS0p2Su_Y8" target="_blank">
    <img src="https://youtu.be/DWSJ-DbVl58?si=Tdz84tgS0p2Su_Y8" alt="Watch our trailer!" width="70%" style="border-radius: 10px;" />
  </a >
  <p><b>▶️ Click here to watch the full video tutorial</b></p>
</div>


The system features 4 collaborative robots handling multiple catering related tasks for an efficient, and safe display of SafeCo's engineering abilities.

* Lynxmotion SES-PRO 900mm 6Dof Modular Robotic Arm Kit: Used for kinematics applications by engineering students, something that is very appealing for future opportunities and experiences. The 3kg payload limit also enables a versatile range in task options.
* FAIRINO FR3 Collaborative Robot 6-axis compact collaborative robot: A 3 kg payload, 622 mm reach and ±0.02 mm repeatability. This robot is especially suited for production environments where space may be limited while delivering precise and repeatable control actions. The FR3 is suitable for companies that may have little to no robotics experience where motion can be taught directly on the robot combined with sequences via an intuitive user interface. This makes it a suitable robot solution accessible for beginner applications with the option of further implementations.
* JAKA MiniCobo: Ultra-lightweight, 6DoF cobot explicitly designed for service, commercial, and hospitality environments. Weighing just 9.4 kg with a 1 kg payload capacity and a 580 mm reach, it offers unique advantages for catering, cafés, and food service automation.
* Universal Robots UR3e: A compact robotic arm with a 3 kg payload and 500 mm reach, making it suitable for precise, automated catering tasks.

# Contributors
* Christina Eugene - FAIRINO FR3
* Anthony Fava - Lynxmotion SES-PRO
* Kaja Katovcic - JAKA MiniCobo 

# Shared workcell
All positions live in `workcell_layout.py` (one world frame). Robots are mounted at `STATIONS[...]` and props are placed relative to that base frame.The tray stops for each station on the conveyor and carries the order out through the light curtain to the collection zone.

Each robot has a station demo that runs on its own and `main_scene.py` runs every integrated robot together. All of them share one planner, one sequencer, one safety controller and one Swift panel.

The integrated Swift panel also provides controlled movable obstacles for collision-avoidance demonstrations, including the JAKA milk pitcher, Lynxmotion cookie jar and FAIRINO FR3 blue crate (drink can storage). Moving an obstacle into a planned path causes the affected robot to check and, where possible, re-plan a collision-free trajectory.

| File | Contents |
| --- | --- |
| `workcell_layout.py` | World frame, conveyor, enclosure, station base poses, safety equipment positions |
| `workcell_scene.py` | Enclosure, conveyor and tray, collection zone, safety equipment (with the risk register), reserved robot footprints |
| `safety_controller.py` | E-stop / light curtain / fault state machine: latched stop, separate RESET and START. Light curtain modelled as beams (Lab 6 ray casting) |
| `collisions.py` | Lab 5 line-plane test on link, tool and payload centre-lines + Lab 6 ellipsoids for the tool and payload, against named + rotatable `RectangularPrism` obstacles |
| `motion_planner.py` | **Shared by every robot**: jtraj for free-space moves and RMRC for constrained Cartesian moves, IK, collision checks, active avoidance. Details changing per robot go in a `PlannerSettings` |
| `task_sequencer.py` | **Shared**: runs every robot and the conveyor frame by frame as 'lanes' synchronised by flags. Re-plans after a stop or when an obstacle moves |
| `cell_runner.py` | **Shared**: Swift set-up, safety panel and real-time loop |
| `teach_pendant.py` | **Shared**: teach/jog GUI (AUTO / MANUAL, joint sliders), pygame joystick jogging with rumble at joint limits, joystick Cross (X) button = physical e-stop |
| `plan_report.py` | **Shared**: Use `--plan` for jtraj vs RMRC comparison and planned-sequence tables |
| `jaka_motion.py` | JAKA planner settings: gripper, side grasp, speeds, gripper/cup collision shapes |
| `coffee_station.py` | JAKA station: bench, cup dispenser, coffee machine, cup, gripper, movable milk pitcher |
| `coffee_task.py` | JAKA job (coffee segments, gripper and brew actions) |
| `ur3e_motion.py` | UR3e planner settings: suction tool, tool-down grasp, speeds, tool/plate collision shapes |
| `ur3e_station.py` | UR3e station: bench, plate dispenser with guide posts, plates, suction tool |
| `ur3e_task.py` | UR3e job (plate segments, suction actions) |
| `lynx_motion.py` | Lynxmotion planner settings: gripper, tool-down grasp, speed, gripper/cookie collision shapes |
| `lynx_station.py` | Lynxmotion station: bench, cookie board, cookies, movable cookie jar |
| `lynx_task.py` | Lynxmotion job (cookie segments, gripper actions) |
| `fairino_fr3.py` | FAIRINO FR3 robot model and DH parameters with 3D mesh configuration |
| `fairino_fr3_motion.py` | FR3 planner settings: top-down gripper, speeds, tool/drink collision geometry |
| `fairino_fr3_station.py` | FR3 station: bench, cold drink, gripper, movable blue obstacle crate |
| `fairino_fr3_task.py` | FR3 job: drink pick, carry, placement and gripper actions |
| `FAIRINO_FR3_Meshes/` | 3D mesh files for the FAIRINO FR3 links |
| `main_scene.py` | Whole cell: UR3e plate, Lynxmotion cookie, FR3 cold drink can, JAKA coffee, then collection |
| `demos_and_debug/` | Each station's demo (`coffee_demo.py`, `ur3e_demo.py`, `lynx_demo.py`, `fairino_fr3_demo.py`) plus test and debug scripts |

```
python main_scene.py --plan                              # every robot's planned sequence, no Swift
python main_scene.py                                     # the whole cell: press START in the side panel
python demos_and_debug/coffee_demo.py --plan             # one station: jtraj vs RMRC table + planned sequence
python demos_and_debug/coffee_demo.py                    # one station in Swift
python demos_and_debug/ur3e_demo.py                      # (--plan works for every demo)
python demos_and_debug/lynx_demo.py
python demos_and_debug/fairino_fr3_demo.py

```

**Teach/jog and e-stops** (Swift side panel in every demo and `main_scene.py`):
- **AUTO/MANUAL** toggles the mode. In MANUAL the task sequence and conveyor pause and the robot chosen in **Robot** is jogged. Switching back to AUTO re-plans the moves in progress from where the arms were left.
- **Joint jog**: one slider per joint (degrees).
- **Cartesian jog (joystick)** (PS4 layout as in Lab 11): left stick = X/Y, Triangle = Z up, Circle = Z down. Rumbles while a jog pushes a joint against its limit or is blocked by a predicted collision.
- **Physical e-stop: joystick Cross (X) button** (pygame button 0), active in AUTO and MANUAL. Unplugging the joystick also e-stops.
- E-stops latch per source (GUI, joystick). Releasing one never restarts or clears another: **RESET**, then **START**. Jogging is blocked while stopped and between RESET and START.
- If the cell faults after jogging (an arm left somewhere it cannot plan out of), stay in MANUAL, RESET, START, jog the arm clear, then switch back to AUTO.

| Robot | File |
| --- | --- |
| UR3e | `ur3e_task.py` |
| Lynxmotion SES-PRO | `lynx_task.py` |
| FAIRINO FR3 | `fairino_fr3_task.py` |
| JAKA MiniCobo | `coffee_task.py` |

