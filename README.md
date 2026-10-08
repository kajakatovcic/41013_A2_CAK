# Presenting: The Smooth Oper-Caterer - 41013 (Industrial Robotics)
SafeCo wants to present the safe, collaborative features of their robotic systems at conferences and meetings through an interactive catering system during lunch/tea breaks. Attendees can experience the complex capabilities of SafeCo’s systems as they are served their choice of food, beverages and accompaniments from the ‘Smooth Oper-Caterer’.

The system features 4 collaborative robots handling multiple catering related tasks for an efficient, and safe display of SafeCo's engineering abilities.

* Lynxmotion SES-PRO 900mm 6Dof Modular Robotic Arm Kit: Used for kinematics applications by engineering students, something that is very appealing for future opportunities and experiences. The 3kg payload limit also enables a versatile range in task options.
* FAIRINO FR3 Collaborative Robot 6-axis compact collaborative robot: A 3 kg payload, 622 mm reach and ±0.02 mm repeatability. This robot is especially suited for production environments where space may be limited while delivering precise and repeatable control actions. The FR3 is suitable for companies that may have little to no robotics experience where motion can be taught directly on the robot combined with sequences via an intuitive user interface. This makes it a suitable robot solution accessible for beginner applications with the option of further implementations.
* JAKA MiniCobo: Ultra-lightweight, 6DoF cobot explicitly designed for service, commercial, and hospitality environments. Weighing just 9.4 kg with a 1 kg payload capacity and a 580 mm reach, it offers unique advantages for catering, cafés, and food service automation.
* OMRON TM5700: A medium-duty robotic arm with a 6 kg payload and 700 mm reach, making it suitable for automated food packaging, end-of-line palletizing, and material handling in food production.

# Contributors
* Christina Eugene - FAIRINO FR3
* Anthony Fava - Lynxmotion SES-PRO
* Kaja Katovcic - JAKA MiniCobo 

# Shared workcell
All positions live in `workcell_layout.py` (one world frame). Mount your robot at `STATIONS[...]` and place your props relative to that base frame, the way `coffee_station.py` does for the JAKA. The tray stops for each station on the conveyor and carries the order out through the light curtain to the collection zone.

| File | Contents |
| --- | --- |
| `workcell_layout.py` | World frame, conveyor, enclosure, station base poses, safety equipment positions |
| `workcell_scene.py` | Enclosure, conveyor and tray, collection zone, safety equipment (with the risk register), reserved robot footprints |
| `safety_controller.py` | E-stop / light curtain / fault state machine: latched stop, separate RESET and START. Light curtain modelled as beams (Lab 6 ray casting) |
| `collisions.py` | Lab 5 line-plane test on link, gripper and cup centre-lines, plus Lab 6 ellipsoids for the gripper and cup, against named, rotatable `RectangularPrism` obstacles |
| `coffee_station.py` | JAKA station: bench, cup dispenser, coffee machine, cup, gripper, movable milk pitcher |
| `jaka_motion.py` | JAKA planning: jtraj for free-space transits, RMRC for constrained Cartesian moves, active avoidance |
| `coffee_task.py` | Frame-by-frame sequencer for the JAKA and the conveyor |
| `coffee_demo.py` | Entry point |

```
python coffee_demo.py --plan                         # no Swift: jtraj vs RMRC table + planned sequence
python coffee_demo.py                                # Swift demo; press START in the side panel
python coffee_demo.py --final-q "60,-40,-100,0,50,0" # add a supplied final joint state (degrees)
```


