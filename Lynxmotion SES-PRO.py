import os
import sys
import trimesh
import numpy as np
import shutil, tempfile
import roboticstoolbox as rtb
from roboticstoolbox import Link, ET, ERobot
from spatialmath import SE3
from spatialgeometry import Cylinder, Cuboid, Mesh

MESH_DIR = os.environ.get("LYNX_MESH_DIR") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "41013_A2_CAK",
    "Lynxmotion_Meshes"
)

BRACKET_LIFT = 0.0318      
BASE_Z = 0.0329          
WITH_GRIPPER = True


def safe_mesh_path(filename):
    src = os.path.join(MESH_DIR, filename)
    dst_dir = os.path.join(tempfile.gettempdir(), "lynx_meshes")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, filename)
    shutil.copy(src, dst)
    return dst

# ------------------------------------------------------------
# MESH MEASUREMENT
# ------------------------------------------------------------
FILES = {"m1": "lss_m1_lo.stl", "s1": "lss_s1_lo.stl", "l1": "lss_l1_lo.stl",
         "tube": "cf_tube_280.stl"}

AXIS_FORCE = {}


def _circularity(V, ia):
    from scipy.spatial import ConvexHull
    p, q = [i for i in range(3) if i != ia]
    h = ConvexHull(V[:, [p, q]])
    return 4 * np.pi * h.volume / h.area ** 2      # 2-D: volume = area, area = perimeter


def _capfrac(m):
    fn, ar = np.asarray(m.face_normals), np.asarray(m.area_faces)
    return [float(ar[np.abs(fn[:, a]) > 0.985].sum() / ar.sum()) for a in range(3)]


def measure(kind):
    m = trimesh.load(os.path.join(MESH_DIR, FILES[kind]))
    V = np.asarray(m.vertices)
    lo, hi = V.min(0), V.max(0)
    ext = hi - lo
    if kind == "tube":
        ia = int(np.argmax(ext))
    elif kind in AXIS_FORCE:
        ia = AXIS_FORCE[kind]
    else:
        scores = [_circularity(V, i) for i in range(3)]
        ia = int(np.argmax(scores))
    p, q = [(ia + 1) % 3, (ia + 2) % 3]
    length = ext[ia]

    from trimesh.nsphere import minimum_nsphere
    tol = 0.02 * length
    cap = V[(V[:, ia] < lo[ia] + tol) | (V[:, ia] > hi[ia] - tol)]
    c2, r_c = minimum_nsphere(cap[:, [p, q]])
    centre = np.zeros(3)
    centre[ia] = 0.5 * (lo[ia] + hi[ia])
    centre[p], centre[q] = c2

    off = V[:, [p, q]] - c2
    dist = np.hypot(off[:, 0], off[:, 1])
    mask = dist > r_c + 0.002
    if mask.sum() >= 3:
        bump_h = float(dist[mask].max() - r_c)
        mo = off[mask].mean(0)
        bi = 0 if abs(mo[0]) >= abs(mo[1]) else 1
        p_axis = [p, q][bi]
        psign = 1.0 if mo[bi] >= 0 else -1.0
        bump_mid = float(0.5 * (V[mask, ia].min() + V[mask, ia].max()))
    else:
        bump_h, p_axis, psign, bump_mid = 0.0, p, 1.0, float(centre[ia])

    span = 0.08 * length
    n_lo = int((V[:, ia] < lo[ia] + span).sum())
    n_hi = int((V[:, ia] > hi[ia] - span).sum())
    hub_sign = 1.0 if n_hi >= n_lo else -1.0   

    return dict(kind=kind, file=FILES[kind], ia=ia, p=p_axis, psign=psign, length=float(length),
                circ=[float(_circularity(V, i)) for i in range(3)],
                caps=_capfrac(m), lo=lo, hi=hi,
                radius=float(r_c), centre=centre, bump_h=bump_h, bump_mid=bump_mid,
                hub_sign=hub_sign, hub_counts=(n_lo, n_hi))


MEAS = {k: measure(k) for k in FILES}


def place(kind, axis, strip, pos=(0, 0, 0)):
    M = MEAS[kind]
    e = np.eye(3)
    ia, p = M["ia"], M["p"]
    ep = M["psign"] * e[p]                
    mesh_basis = np.column_stack([e[ia], ep, np.cross(e[ia], ep)])
    a = np.array(axis, float); a /= np.linalg.norm(a)
    s = np.array(strip, float); s /= np.linalg.norm(s)
    world_basis = np.column_stack([a, s, np.cross(a, s)])
    R = world_basis @ mesh_basis.T
    return SE3(*pos) * SE3.Rt(R, -R @ M["centre"])

# ------------------------------------------------------------
# DIMENSIONS (all metres)
# ------------------------------------------------------------
TUBE_LEN = MEAS["tube"]["length"]     
TUBE_R = MEAS["tube"]["radius"]
EMB = 0.02                
OV = 0.02                 
SH_SINK = -0.012            
SHOULDER_STRIP_TO_TUBE = True   
                               
ELBOW_SPIN_DEG = -90          
                              
ELBOW_ROLL_DEG = 270           
                               
J4_SPIN_DEG = -90              
J4_FLIP_DEG = 180              

J5_SPIN_DEG = -45             
J5_FLIP_DEG = 180              
J5_TURN_DEG = 90              
                            
J5_FLIP_ABOUT = "z"           

J6_SPIN_DEG = 90              
J6_SIDE = -1                  
ELBOW_FLIP = False             
J4_GAP = 0.002                

LEN_M1, LEN_S1, LEN_L1 = (MEAS[k]["length"] for k in ("m1", "s1", "l1"))
R_M1, R_S1, R_L1 = (MEAS[k]["radius"] for k in ("m1", "s1", "l1"))
CLAMP = 0.015            

WAIST_DX = 0.0            
WAIST_DY = 0.0
SHOULDER_FLIP = False      
SEAT = 0.01               
_Mw = MEAS["m1"]
WAIST_PANEL_Z = LEN_M1 / 2 + (_Mw["bump_mid"] - _Mw["centre"][_Mw["ia"]])
D1 = WAIST_PANEL_Z                       
SH_FACE_Y = WAIST_DY + R_M1 + _Mw["bump_h"] - SEAT  
SH_YC = SH_FACE_Y + LEN_M1 / 2           
SH_END = SH_YC + LEN_M1 / 2              
Y_UP = SH_YC                              
SH_TUBE_Z0 = R_M1 + (_Mw["bump_h"] if SHOULDER_STRIP_TO_TUBE else 0.0) - SH_SINK

_Ms = MEAS["s1"]
def _rz(d):
    c, s_ = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
def _rx(d):
    c, s_ = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
def _ry(d):
    c, s_ = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
_EL_R = _ry(ELBOW_ROLL_DEG) @ _rz(ELBOW_SPIN_DEG)
EL_AXIS = tuple(_EL_R @ np.array([-_Ms["hub_sign"] * (-1 if ELBOW_FLIP else 1), 0, 0]))
EL_STRIP = tuple(_EL_R @ np.array([0, 0, -1]))
_J4_R = _ry(J4_FLIP_DEG) @ _rz(J4_SPIN_DEG)
J4_AXIS = tuple(_J4_R @ np.array([0, 0, 1]))
J4_STRIP = tuple(_J4_R @ np.array([-1, 0, 0]))
_J5_R = _ry(J5_TURN_DEG) @ {"z": _rz, "x": _rx}[J5_FLIP_ABOUT](J5_FLIP_DEG) @ _ry(J5_SPIN_DEG)
J5_AXIS = tuple(_J5_R @ np.array([0, 1, 0]))
J5_STRIP = tuple(_J5_R @ np.array([-1, 0, 0]))
J6_STRIP = tuple(_rz(J6_SPIN_DEG) @ np.array([-1, 0, 0]))

EL_TUBE_Z0 = R_S1 + (_Ms["bump_h"] if EL_STRIP[2] < -0.5 else 0.0) - SH_SINK
L_UPPER = SH_TUBE_Z0 + TUBE_LEN + EL_TUBE_Z0
ELB_BODY_YC = 0.0                     

Y_FORE = -max(LEN_S1 / 2 + R_S1 - 0.003,          
              TUBE_R + R_S1 + J4_GAP)              
TUBE_CLEARANCE = abs(Y_FORE) - R_S1 - TUBE_R   

L_J5 = LEN_S1 / 2 + TUBE_LEN

J6_SIDE_X = 0
J6_OFFSET = J6_SIDE_X * (R_L1 + R_L1 - CLAMP)
J6_OFFSET_Y = 0.028 

TOOL_Z = -(LEN_L1 / 2 + 0.085)

QLIM = [-np.pi, np.pi]

X_, Y_, Z_ = (1, 0, 0), (0, 1, 0), (0, 0, 1)

class LynxmotionSESPro(ERobot):

    def __init__(self):

        l1 = Link(ET.Rz(qlim=QLIM), name="waist")
        l2 = Link(ET.tz(D1) * ET.Ry(qlim=QLIM), name="shoulder", parent=l1)
        l3 = Link(ET.ty(Y_UP) * ET.tz(L_UPPER) * ET.Ry(qlim=QLIM),
                  name="elbow", parent=l2)
        l4 = Link(ET.ty(Y_FORE) * ET.Rz(qlim=QLIM),
                  name="wrist_roll", parent=l3)
        l5 = Link(ET.tz(L_J5) * ET.Ry(qlim=QLIM), name="wrist_pitch", parent=l4)
        l6 = Link(
            ET.tx(J6_OFFSET) * ET.ty(J6_OFFSET_Y) * ET.Rz(qlim=QLIM),
            name="wrist_yaw",
            parent=l5
        )

        super().__init__(
            [l1, l2, l3, l4, l5, l6],
            name="LynxmotionSESPro",
            manufacturer="Lynxmotion",
            tool=SE3.Tz(TOOL_Z),
        )

        self.base = SE3.Tz(BASE_Z)
        deg = np.pi / 180
        self.qr = np.array([0, -35 * deg, 75 * deg, 0, -40 * deg, 0])
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_visuals(self):

        HOUSING = (0.22, 0.22, 0.25, 1)
        TUBE_COL = (0.08, 0.08, 0.09, 1)
        BASE_COL = (0.10, 0.10, 0.11, 1)
        GRIP = (0.60, 0.60, 0.63, 1)

        base_plate = Mesh(filename=safe_mesh_path("base_bracket.stl"),
                          scale=[1, 1, 1], color=BASE_COL,
                          pose=SE3(0, 0, BRACKET_LIFT))

        def actuator(kind):
            return Mesh(filename=safe_mesh_path(FILES[kind]),
                        scale=[1, 1, 1], color=HOUSING)

        def tube():
            return Mesh(filename=safe_mesh_path(FILES["tube"]),
                        scale=[1, 1, 1], color=TUBE_COL)

        S = (-1, 0, 0)                      
        SH_STRIP = Z_ if SHOULDER_STRIP_TO_TUBE else S  

        tube_z = SH_TUBE_Z0 + TUBE_LEN / 2  
        tube2_z = LEN_S1 / 2 - EMB + TUBE_LEN / 2   

        parts = [
          
            (-1, actuator("m1"), place("m1", Z_, (0, 1, 0), (WAIST_DX, WAIST_DY, LEN_M1 / 2))),
            (1, actuator("m1"), place("m1",
                (0, MEAS["m1"]["hub_sign"] * (-1 if SHOULDER_FLIP else 1), 0), SH_STRIP, (0, SH_YC, 0))),
            (1, tube(), place("tube", Z_, S, (0, Y_UP, tube_z))),

            (2, actuator("s1"), place("s1", EL_AXIS, EL_STRIP, (0, ELB_BODY_YC, 0))),

            (3, actuator("s1"), place("s1", J4_AXIS, J4_STRIP)),
            (3, tube(), place("tube", Z_, S, (0, 0, tube2_z))),

            (4, actuator("l1"), place("l1", J5_AXIS, J5_STRIP)),

            (5, actuator("l1"), place("l1", Z_, J6_STRIP)),
        ]

        if WITH_GRIPPER:
            z0 = -LEN_L1 / 2

            parts += [
                (5, Cylinder(radius=0.02, length=0.06, color=GRIP),
                SE3(0, 0, z0 - 0.03)),

                (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
                SE3(0.02, 0, z0 - 0.085)),

                (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
                SE3(-0.02, 0, z0 - 0.085)),
            ]
        return base_plate, parts

    def pose_visuals(self, q, parts):
        link_frames = self.fkine_all(q)
        for link_idk, shape, offset in parts:
            if link_idk < 0:
                frame = self.base
            else:
                frame = link_frames[link_idk + 1]
            shape.T = (frame * offset).A


def run_gap_report(n_random=60):

    from scipy.spatial import cKDTree
    r = LynxmotionSESPro()
    _, parts = r.link_visuals()
    files = []
    for _, shape, _off in parts:
        files.append(getattr(shape, "filename", None))
    kinds = ["m1", "m1", "tube", "s1", "s1", "tube", "l1", "l1"]
    names = ["waist", "shoulder", "tube1", "elbow", "roll", "tube2", "pitch", "yaw"]
    samples = [trimesh.load(os.path.join(MESH_DIR, FILES[k])).sample(6000) for k in kinds]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    worst = np.zeros(len(pairs)); tube_min = 9.0
    rng = np.random.default_rng(0)
    for k in range(n_random):
        q = r.qz if k == 0 else (r.qr if k == 1 else rng.uniform(-np.pi, np.pi, 6))
        r.pose_visuals(q, parts)
        pts = [smp @ shp.T[:3, :3].T + shp.T[:3, 3]
               for smp, (_, shp, _o) in zip(samples, parts[:8])]
        for j, (a, b) in enumerate(pairs):
            worst[j] = max(worst[j], cKDTree(pts[a]).query(pts[b])[0].min())
        tube_min = min(tube_min, cKDTree(pts[2]).query(pts[4])[0].min(),
                       cKDTree(pts[2]).query(pts[5])[0].min())
    print(f"tube1 vs roll-actuator plane clearance by design: {TUBE_CLEARANCE*1000:.1f} mm")
    print(f"closest tube1<->(roll actuator, tube2) over {n_random} poses: {tube_min*1000:.1f} mm  (must be > 0)")
    for (a, b), w in zip(pairs, worst):
        print(f"  {names[a]:8s}-{names[b]:8s} worst gap {w*1000:5.1f} mm")


def run_check():
    for name in ("base_bracket.stl", "cf_tube_280.stl", "cf_clamp.stl",
                 "lss_m1_lo.stl", "lss_s1_lo.stl", "lss_l1_lo.stl"):
        try:
            m = trimesh.load(os.path.join(MESH_DIR, name))
            lo, hi = m.bounds
            print(f"{name:18s} x[{lo[0]:+.3f},{hi[0]:+.3f}] "
                  f"y[{lo[1]:+.3f},{hi[1]:+.3f}] z[{lo[2]:+.3f},{hi[2]:+.3f}]")
        except Exception as e:
            print(name, "ERROR", e)
    print("MEASURED (axis 0/1/2 = mesh X/Y/Z):")
    for k, M in MEAS.items():
        print(f"  {k:4s} axis={M['ia']} length={M['length']:.4f} radius={M['radius']:.4f} "
              f"circ(X,Y,Z)={np.round(M['circ'], 2)} flat-caps(X,Y,Z)={np.round(M['caps'], 2)} "
              f"bump={M['bump_h']*1000:.1f}mm")
        print(f"    hub_sign={M['hub_sign']:+.0f} end-cap vertex counts (lo, hi)={M['hub_counts']}")
    r = LynxmotionSESPro()
    print(f"waist holes-panel height z={WAIST_PANEL_Z:.4f} (base frame)  shoulder end face y={SH_FACE_Y:.4f}")
    print(f"D1={D1:.4f}  upper-arm plane y={Y_UP:.4f}  forearm plane y(elbow)={Y_FORE:.4f}")
    print(f"J6_OFFSET (J5 side -> J6 centre, along J5 frame X) = {J6_OFFSET:.4f}")
    for label, q in (("qz", r.qz), ("qr(photo)", r.qr)):
        T = r.fkine_all(q)
        print(label)
        for i, nm in enumerate(["base", "waist", "shoulder", "elbow", "wrist_roll",
                                "wrist_pitch", "wrist_yaw"]):
            print(f"  {nm:12s} origin = {np.round(T[i].t, 3)}")
        print("  tool tip     =", np.round(r.fkine(q).t, 3))
    print()
    run_gap_report()


def candidate_pose(kind, ia, x0, y0):
    M = MEAS[kind]
    e = np.eye(3); p = (ia + 1) % 3
    mesh_basis = np.column_stack([e[ia], e[p], np.cross(e[ia], e[p])])
    world_basis = np.column_stack([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R = world_basis @ mesh_basis.T
    c = 0.5 * (M["lo"] + M["hi"])
    return SE3(x0, y0, 0.1) * SE3.Rt(R, -R @ c)


def run_candidates():
    from swift import Swift
    env = Swift()
    env.launch(realtime=True)
    tiles = [(1, 0.2, 0.2, 1), (0.2, 1, 0.2, 1), (0.2, 0.2, 1, 1)]
    for row, k in enumerate(("m1", "s1", "l1")):
        for col, ia in enumerate((0, 1, 2)):
            x0, y0 = 0.3 * col, 0.3 * row
            env.add(Cuboid(scale=[0.22, 0.22, 0.004], color=tiles[col], pose=SE3(x0, y0, -0.002)))
            env.add(Mesh(filename=safe_mesh_path(FILES[k]), scale=[1, 1, 1],
                         color=(0.55, 0.55, 0.6, 1), pose=candidate_pose(k, ia, x0, y0)))
    print("candidates: columns left->right = mesh axis X (red tile), Y (green), Z (blue);")
    print("            rows front->back = m1, s1, l1. Which column stands upright like a can?")
    env.step()
    env.hold()


def run_axes():
    from swift import Swift
    env = Swift()
    env.launch(realtime=True)
    for i, k in enumerate(("m1", "s1", "l1")):
        x0 = 0.35 * i
        env.add(Mesh(filename=safe_mesh_path(FILES[k]), scale=[1, 1, 1],
                     color=(0.4, 0.4, 0.45, 1), pose=SE3(x0, 0, 0)))
        env.add(Cuboid(scale=[0.3, 0.004, 0.004], color=(1, 0, 0, 1), pose=SE3(x0, 0, 0)))
        env.add(Cuboid(scale=[0.004, 0.3, 0.004], color=(0, 1, 0, 1), pose=SE3(x0, 0, 0)))
        env.add(Cuboid(scale=[0.004, 0.004, 0.3], color=(0, 0, 1, 1), pose=SE3(x0, 0, 0.0)))
    env.step()
    env.hold()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "candidates":
        run_candidates()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "axes":
        run_axes()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        run_check()
        sys.exit(0)

    from swift import Swift

    robot = LynxmotionSESPro()
    print(robot)
    for k, M in MEAS.items():
        print(f"measured {k:4s}: cylinder axis = mesh {'XYZ'[M['ia']]}, "
              f"length {M['length']:.3f} m, radius {M['radius']:.3f} m")
        if k != "tube" and int(np.argmax(M["caps"])) != M["ia"]:
            print(f"   WARNING: {k}: detectors disagree (circularity says {'XYZ'[M['ia']]}, "
                  f"flat caps say {'XYZ'[int(np.argmax(M['caps']))]}) - run the 'candidates' mode")

    base_plate, parts = robot.link_visuals()

    env = Swift()
    env.launch(realtime=True)
    env.add(base_plate)
    for _, shape, _ in parts:
        env.add(shape)

    if len(sys.argv) > 1 and sys.argv[1] == "static":
        robot.q = robot.qr
        robot.pose_visuals(robot.q, parts)
        env.step()
        env.hold()
        sys.exit(0)

    robot.q = robot.qz
    robot.pose_visuals(robot.q, parts)
    env.step()

    d = np.pi / 180
    waypoints = [
        robot.qz,
        robot.qr,
        np.array([60, -20, 50, 90, -30, 90]) * d, 
        np.array([-60, 20, 70, -60, 45, -90]) * d,  
        np.array([0, -50, 100, 0, 60, 180]) * d,  
        robot.qr,
    ]

    for i in range(len(waypoints) - 1):
        traj = rtb.jtraj(waypoints[i], waypoints[i + 1], 60)
        for q in traj.q:
            robot.q = q
            robot.pose_visuals(q, parts)
            env.step(0.02)

    env.hold()

