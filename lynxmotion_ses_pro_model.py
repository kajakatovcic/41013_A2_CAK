import os                                           # file folders
import shutil, tempfile                             # copy files into a temporary folder
import numpy as np                                  # calculator
import trimesh                                      # 3D model reader
from trimesh.nsphere import minimum_nsphere         # find the smallest circle which fits around a bunch of points
from scipy.spatial import ConvexHull                # 'rubber band' around the points of the circle
from roboticstoolbox import Link, ET, ERobot        # link 'bone', how the bone moves, and the whole 'skeleton'
import roboticstoolbox as rtb                       # DH links
from spatialmath import SE3                         # position and direction GPS
from spatialgeometry import Cylinder, Cuboid, Mesh  # shapes for modelling

MESH_DIR = os.environ.get("LYNX_MESH_DIR") or os.path.join( # where the parts are located
    os.path.dirname(os.path.abspath(__file__)), "Lynxmotion_Meshes")   # next to this file, whatever the repo folder is called

# ============================================================
# TUNING
# ============================================================
WITH_GRIPPER = True       # toggle gripper on and off
BRACKET_LIFT = 0.0318     # base bracket height from the floor
BASE_Z = 0.0329           # robot base frame height from the floor
WAIST_DX = 0.0            # waist actuator translates across the x-axis
WAIST_DY = 0.0            # waist actuator translates across the y-axis
# How far each part is to each other 
SEAT = 0.002              # shoulder end face -> waist panel
SH_SINK = 0.002           # tube 1 -> shoulder
J4_SINK = 0.002           # tube 2 -> roll actuator (J4)
J6_SINK = 0.002           # yaw actuator (J6) -> pitch actuator (J5)
TUBE1_END_TO_AXIS = 0.002 # tube 1's top end stops this far below the elbow axis
TUBE2_END_TO_AXIS = 0.002 # tube 2's top end stops this far below the J5 axis
J4_GAP = 0.002            # minimum clearance between J4 and tube 1
TUBE_OD_MAX = 0.070       # real tube outer diameter (SKU CF-70x67-280.5)
TUBE_FIT = 0.92           # tube radius vs the narrowest housing body it plugs into

# Joint limits (rad), waist +-360 deg, the rest +-180 deg
QLIMS = [[-2 * np.pi, 2 * np.pi]] + [[-np.pi, np.pi]] * 5

# ============================================================
# MESHES
# ============================================================
FILES = {"s1": "lss_s1_lo.stl", "m1": "lss_m1_lo.stl", "l1": "lss_l1_lo.stl", # small, medium, and large actuators
         "tube": "cf_tube_280.stl"}
Z_ = (0, 0, 1)

def safe_mesh_path(filename): # avoid trouble with folder names and locked files
    dst_dir = os.path.join(tempfile.gettempdir(), "lynx_meshes")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, filename)
    shutil.copy(os.path.join(MESH_DIR, filename), dst)
    return dst

def _circularity(V, ia):                        # ignore direction ia and look at the shadow of the part from the other two, then ConvexHull around shadow
    p, q = [i for i in range(3) if i != ia] 
    h = ConvexHull(V[:, [p, q]])
    return 4 * np.pi * h.volume / h.area ** 2   # compute roundness score where 1.0 is perfect

def measure(kind):
    V = np.asarray(trimesh.load(os.path.join(MESH_DIR, FILES[kind])).vertices)
    lo, hi = V.min(0), V.max(0) # load vertices
    ext = hi - lo
    ia = int(np.argmax(ext)) if kind == "tube" else int(np.argmax([_circularity(V, i) for i in range(3)]))  # find long axis
    p, q = (ia + 1) % 3, (ia + 2) % 3                                                                       # other two directions of the actuator
    length = ext[ia]                                                                                        # length of the part
    tol = 0.02 * length                                                                                     # true cylinder centre + radius from the two end caps (the connector bump sits on the side)
    cap = V[(V[:, ia] < lo[ia] + tol) | (V[:, ia] > hi[ia] - tol)]
    c2, r_c = minimum_nsphere(cap[:, [p, q]])
    centre = np.zeros(3)
    centre[ia] = 0.5 * (lo[ia] + hi[ia])                                                                    # built from circle centre plus halfway point along the length
    centre[p], centre[q] = c2
    off = V[:, [p, q]] - c2                                                                                 # connector bump = anything sticking out of that cylinder
    dist = np.hypot(off[:, 0], off[:, 1])
    mask = dist > r_c + 0.002
    if mask.sum() >= 3:
        bump_h = float(dist[mask].max() - r_c)
        mo = off[mask].mean(0)
        bi = 0 if abs(mo[0]) >= abs(mo[1]) else 1
        p_axis, psign = [p, q][bi], (1.0 if mo[bi] >= 0 else -1.0)
        bump_mid = float(0.5 * (V[mask, ia].min() + V[mask, ia].max()))
    else:                                                                                                   # counts vertices and finds hub end
        bump_h, p_axis, psign, bump_mid = 0.0, p, 1.0, float(centre[ia])                                    # hub end = the finely tessellated end (the face with the holes)
    span = 0.08 * length
    hub_sign = 1.0 if (V[:, ia] > hi[ia] - span).sum() >= (V[:, ia] < lo[ia] + span).sum() else -1.0
    return dict(ia=ia, p=p_axis, psign=psign, length=float(length), radius=float(r_c),
                centre=centre, bump_h=bump_h, bump_mid=bump_mid, hub_sign=hub_sign, verts=V)

MEAS = {k: measure(k) for k in FILES}   # regarding length and radius of each actuator, and the tube length to find shortcuts
LEN_M1, LEN_S1, LEN_L1 = (MEAS[k]["length"] for k in ("m1", "s1", "l1"))
R_M1, R_S1, R_L1 = (MEAS[k]["radius"] for k in ("m1", "s1", "l1"))
TUBE_LEN = MEAS["tube"]["length"]

def place(kind, axis, strip, pos=(0, 0, 0)):
    M = MEAS[kind]
    e = np.eye(3)
    ep = M["psign"] * e[M["p"]]
    mesh_basis = np.column_stack([e[M["ia"]], ep, np.cross(e[M["ia"]], ep)])    # coordinate system for the mesh and a target
    a = np.array(axis, float); a /= np.linalg.norm(a)                           # where the long axis should point
    s = np.array(strip, float); s /= np.linalg.norm(s)                          # where the connector bump should face
    R = np.column_stack([a, s, np.cross(a, s)]) @ mesh_basis.T
    return SE3(*pos) * SE3.Rt(R, -R @ M["centre"])                              # move part's own center to the origin and spins around the middle

def _pts(kind, pose): # gives the part’s corner points after it’s been moved and rotated
    return MEAS[kind]["verts"] @ pose.R.T + pose.t

def _surface(V, axis, sign, mask, bin_=0.0005): # finds where a flat face is located
    if int(mask.sum()) < 8:
        return None
    v = sign * V[mask, axis]
    v = v[v >= np.percentile(v, 60)]
    bins = np.floor((v - v.min()) / bin_).astype(int)
    counts = np.bincount(bins)
    best = len(counts) - 1 - int(np.argmax(counts[::-1]))
    return sign * float(v[bins == best].mean())

def _housing_offset(V, axis, length, R): # checks whether the real round motor body is off-center compared with the bounding box
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    u = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(a, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u); v = np.cross(a, u)
    t, x, y = V @ a, V @ u, V @ v
    r = np.hypot(x, y)
    shell = (r > 0.85 * R) & (r < 1.04 * R)
    mid = shell & (np.abs(t) < 0.3 * length)
    if int(mid.sum()) < 12 or int(shell.sum()) < 12:
        return np.zeros(3)
    A = np.c_[2 * x[mid], 2 * y[mid], np.ones(int(mid.sum()))]
    cx, cy, _ = np.linalg.lstsq(A, x[mid] ** 2 + y[mid] ** 2, rcond=None)[0]
    lo_t, hi_t = np.percentile(t[shell], [2, 98])
    off = cx * u + cy * v + 0.5 * (lo_t + hi_t) * a
    return off if np.linalg.norm(off) < 0.3 * R else np.zeros(3)

# ============================================================
# SLIM TUBES
# ============================================================
def _body_radius(kind):                                             # measures the real radius of each motor’s round body
    M = MEAS[kind]; ia = M["ia"]; c = M["centre"]
    mesh = trimesh.load(os.path.join(MESH_DIR, FILES[kind]))
    try:
        S, _ = trimesh.sample.sample_surface(mesh, 60000, seed=0)   # samples 60,000 random points on the surface
    except TypeError:                                 
        np.random.seed(0)
        S, _ = trimesh.sample.sample_surface(mesh, 60000)
    p, q = [i for i in range(3) if i != ia]
    mid = np.abs(S[:, ia] - c[ia]) < 0.15 * M["length"]
    r = np.hypot(S[mid, p] - c[p], S[mid, q] - c[q])
    r = r[r > 0.5 * M["radius"]]
    hist, edges = np.histogram(r, bins=80)                          # keeps the ones in the middle section, computes their distance from the center, and finds the most common radius with a histogram
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))

_R_BODY = {k: _body_radius(k) for k in ("m1", "s1", "l1")}
_TUBE_SCALE = min(1.0, TUBE_FIT * min(_R_BODY.values()) / MEAS["tube"]["radius"],
                  0.5 * TUBE_OD_MAX / MEAS["tube"]["radius"])
TUBE_R = MEAS["tube"]["radius"] * _TUBE_SCALE

def _slim_tube_path():
    M = MEAS["tube"]
    S = np.full(3, _TUBE_SCALE); S[M["ia"]] = 1.0
    m = trimesh.load(os.path.join(MESH_DIR, FILES["tube"]))
    m.vertices = (np.asarray(m.vertices) - M["centre"]) * S + M["centre"]
    dst_dir = os.path.join(tempfile.gettempdir(), "lynx_meshes")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "cf_tube_slim.stl")
    m.export(dst)
    return dst

SLIM_TUBE_PATH = _slim_tube_path()

# ============================================================
# ORIENTATIONS
# ============================================================
SH_AXIS,  SH_STRIP = (0, MEAS["m1"]["hub_sign"], 0), (0, 0, 1)      # shoulder (hub end -> +Y, panel faces tube)
EL_AXIS,  EL_STRIP = (0, MEAS["s1"]["hub_sign"], 0), (0, 0, -1)     # elbow
J4_AXIS,  J4_STRIP = (0, 0, -1), (0, 1, 0)                          # wrist roll
J5_AXIS,  J5_STRIP = (0, -1, 0), (0, 0, -1)                         # wrist pitch
J6_AXIS,  J6_STRIP = Z_,         (0, -1, 0)                         # wrist yaw

# ============================================================
# GEOMETRY
# ============================================================
# J1 -> J2: shoulder's end face meets the waist's connector panel (holes side, facing +Y)
_Mw = MEAS["m1"]
D1 = LEN_M1 / 2 + (_Mw["bump_mid"] - _Mw["centre"][_Mw["ia"]])      # shoulder axis height = panel height
_Vw = _pts("m1", place("m1", Z_, (0, 1, 0), (WAIST_DX, WAIST_DY, LEN_M1 / 2)))
_mw = ((np.abs(_Vw[:, 0] - WAIST_DX) < 0.5 * R_M1) & (np.abs(_Vw[:, 2] - D1) < 0.5 * R_M1)
       & (_Vw[:, 1] > WAIST_DY))
_panel_y = _surface(_Vw, 1, +1, _mw)
WAIST_PANEL_Y = _panel_y if _panel_y is not None else WAIST_DY + R_M1 + _Mw["bump_h"]
SH_YC = WAIST_PANEL_Y - SEAT + LEN_M1 / 2     # shoulder centre along its axis
Y_UP = SH_YC                                  # tube 1 sits in the middle plane of the shoulder

# J2 -> tube 1: tube 1's flat end meets the shoulder's top surface
_Vs = _pts("m1", place("m1", SH_AXIS, SH_STRIP, (0, SH_YC, 0)))
_ms = ((np.abs(_Vs[:, 0]) < 0.6 * TUBE_R) & (np.abs(_Vs[:, 1] - SH_YC) < 0.6 * TUBE_R) & (_Vs[:, 2] > 0))
_top = _surface(_Vs, 2, +1, _ms)
SH_TUBE_Z0 = (_top if _top is not None else R_M1 + _Mw["bump_h"]) - SH_SINK   # tube 1 starts here

# tube 1 -> J3: centre the elbow's real housing on tube 1; the elbow axis sits just above tube 1's end
_d_el = _housing_offset(_pts("s1", place("s1", EL_AXIS, EL_STRIP)), EL_AXIS, LEN_S1, R_S1)
EL_POS = tuple(-_d_el)
L_UPPER = SH_TUBE_Z0 + TUBE_LEN + TUBE1_END_TO_AXIS

# J3 -> J4: roll actuator bolted to the elbow's -Y end face, but clear of tube 1
Y_FORE = -max(LEN_S1 / 2 + R_S1 - 0.003 - EL_POS[1], TUBE_R + R_S1 + J4_GAP)

# J4 -> tube 2 -> J5: tube 2 grows out of J4's top face; the J5 axis sits just above tube 2's end
_Vj4 = _pts("s1", place("s1", J4_AXIS, J4_STRIP))
_m4 = ((np.abs(_Vj4[:, 0]) < 0.9 * TUBE_R) & (np.abs(_Vj4[:, 1]) < 0.9 * TUBE_R) & (_Vj4[:, 2] > 0))
_j4t = _surface(_Vj4, 2, +1, _m4)
TUBE2_Z0 = (_j4t if _j4t is not None else LEN_S1 / 2) - J4_SINK
L_J5 = TUBE2_Z0 + TUBE_LEN + TUBE2_END_TO_AXIS

# J5 -> J6: J6's side face meets J5's end face
_Vj5 = _pts("l1", place("l1", J5_AXIS, J5_STRIP))
_mj = ((np.abs(_Vj5[:, 0]) < 0.9 * R_L1) & (np.abs(_Vj5[:, 2]) < 0.9 * R_L1) & (_Vj5[:, 1] > 0))
_j5e = _surface(_Vj5, 1, +1, _mj)
J5_END_Y = _j5e if _j5e is not None else LEN_L1 / 2
_Vj6 = _pts("l1", place("l1", J6_AXIS, J6_STRIP))
_mj6 = ((np.abs(_Vj6[:, 0]) < 0.9 * R_L1) & (np.abs(_Vj6[:, 2]) < 0.4 * LEN_L1) & (_Vj6[:, 1] < 0))
_j6f = _surface(_Vj6, 1, -1, _mj6)
J6_FACE_Y = _j6f if _j6f is not None else -(R_L1 + (MEAS["l1"]["bump_h"] if J6_STRIP[1] < -0.5 else 0.0))
J6_OFFSET_Y = J5_END_Y - J6_SINK - J6_FACE_Y

TOOL_Z = -(LEN_L1 / 2 + 0.085)               # gripper hangs below J6

# ============================================================
# THE ROBOT
# ============================================================
class LynxmotionSESProET(ERobot):
    """OLD joint-chain model. Reference only, used by check_dh_conversion.py. Delete once the check passes."""

    def __init__(self, mount=None):
        self.mount = SE3() if mount is None else mount
        l1 = Link(ET.Rz(qlim=QLIMS[0]), name="waist")
        l2 = Link(ET.tz(D1) * ET.Ry(qlim=QLIMS[1]), name="shoulder", parent=l1)
        l3 = Link(ET.ty(Y_UP) * ET.tz(L_UPPER) * ET.Ry(qlim=QLIMS[2]), name="elbow", parent=l2)
        l4 = Link(ET.ty(Y_FORE) * ET.Rz(qlim=QLIMS[3]), name="wrist_roll", parent=l3)
        l5 = Link(ET.tz(L_J5) * ET.Ry(qlim=QLIMS[4]), name="wrist_pitch", parent=l4)
        l6 = Link(ET.ty(J6_OFFSET_Y) * ET.Rz(qlim=QLIMS[5]), name="wrist_yaw", parent=l5)

        super().__init__([l1, l2, l3, l4, l5, l6], name="LynxmotionSESPro",
                         manufacturer="Lynxmotion", tool=SE3.Tz(TOOL_Z))

        self.base = self.mount * SE3.Tz(BASE_Z)
        deg = np.pi / 180
        # qr matches the product photo: upper arm back, forearm forward, wrist pitched back
        self.qr = np.array([0, -35 * deg, 75 * deg, 0, -40 * deg, 0])
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    def link_visuals(self):
        HOUSING, TUBE_COL = (0.22, 0.22, 0.25, 1), (0.08, 0.08, 0.09, 1)
        BASE_COL, GRIP = (0.10, 0.10, 0.11, 1), (0.60, 0.60, 0.63, 1)

        base_plate = Mesh(filename=safe_mesh_path("base_bracket.stl"), scale=[1, 1, 1],
                          color=BASE_COL, pose=self.mount * SE3(0, 0, BRACKET_LIFT))

        def actuator(kind):
            return Mesh(filename=safe_mesh_path(FILES[kind]), scale=[1, 1, 1], color=HOUSING)

        def tube():
            return Mesh(filename=SLIM_TUBE_PATH, scale=[1, 1, 1], color=TUBE_COL)

        S = (-1, 0, 0)
        tube1_z = SH_TUBE_Z0 + TUBE_LEN / 2          # tube meshes are centred, axis along Z
        tube2_z = TUBE2_Z0 + TUBE_LEN / 2

        # (frame index, shape, pose in that frame): frame k = link k+1 after its joint; -1 = base
        parts = [
            (0, actuator("m1"), place("m1", Z_, (0, 1, 0), (WAIST_DX, WAIST_DY, LEN_M1 / 2))),   # J1 waist
            (1, actuator("m1"), place("m1", SH_AXIS, SH_STRIP, (0, SH_YC, 0))),                  # J2 shoulder
            (1, tube(), place("tube", Z_, S, (0, Y_UP, tube1_z))),                               # tube 1
            (1, actuator("s1"), SE3(0, Y_UP, L_UPPER) * place("s1", EL_AXIS, EL_STRIP, EL_POS)), # J3 elbow
            (2, actuator("s1"), SE3(0, Y_FORE, 0) * place("s1", J4_AXIS, J4_STRIP)),             # J4 roll
            (3, tube(), place("tube", Z_, S, (0, 0, tube2_z))),                                  # tube 2
            (3, actuator("l1"), SE3(0, 0, L_J5) * place("l1", J5_AXIS, J5_STRIP)),               # J5 pitch
            (4, actuator("l1"), SE3(0, J6_OFFSET_Y, 0) * place("l1", J6_AXIS, J6_STRIP)),        # J6 yaw
        ]

        if WITH_GRIPPER:
            z0 = -LEN_L1 / 2
            parts += [
                (5, Cylinder(radius=0.02, length=0.06, color=GRIP), SE3(0, 0, z0 - 0.03)),
                (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP), SE3(0.02, 0, z0 - 0.085)),
                (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP), SE3(-0.02, 0, z0 - 0.085)),
            ]
        return base_plate, parts

    def tcp(self, q):
        return self.fkine_all(np.clip(q, self.qlim[0], self.qlim[1]))[-1] * SE3.Tz(TOOL_Z)

    def pose_visuals(self, q, parts):
        q = np.clip(q, self.qlim[0], self.qlim[1])      
        link_frames = self.fkine_all(q)
        for link_idk, shape, offset in parts:
            frame = self.base if link_idk < 0 else link_frames[link_idk + 1]
            shape.T = (frame * offset).A

# ============================================================
# THE ROBOT (DH): the one the cell uses
# ============================================================
_H = np.pi / 2

def _dh_links():
    """Standard DH (Rz(q+offset) Tz(d) Tx(a) Rx(alpha)). Gives exactly the same end-effector
    pose as the old joint-chain model (checked by check_dh_conversion.py)."""
    return [
        rtb.RevoluteDH(d=D1,          a=0,       alpha=-_H, offset=0,   qlim=QLIMS[0]),
        rtb.RevoluteDH(d=Y_UP,        a=L_UPPER, alpha=0,   offset=-_H, qlim=QLIMS[1]),
        rtb.RevoluteDH(d=Y_FORE,      a=0,       alpha=_H,  offset=_H,  qlim=QLIMS[2]),
        rtb.RevoluteDH(d=L_J5,        a=0,       alpha=-_H, offset=0,   qlim=QLIMS[3]),
        rtb.RevoluteDH(d=J6_OFFSET_Y, a=0,       alpha=_H,  offset=0,   qlim=QLIMS[4]),
        rtb.RevoluteDH(d=0,           a=0,       alpha=0,   offset=0,   qlim=QLIMS[5]),
    ]


class LynxmotionSESPro(rtb.DHRobot):

    def __init__(self, mount=None):
        self.mount = SE3() if mount is None else mount
        base = self.mount * SE3.Tz(BASE_Z)
        super().__init__(_dh_links(), name="LynxmotionSESPro", base=base)
        self.base = base
        deg = np.pi / 180
        # qr matches the product photo: upper arm back, forearm forward, wrist pitched back
        self.qr = np.array([0, -35 * deg, 75 * deg, 0, -40 * deg, 0])
        self.qz = np.zeros(6)
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)

    # The DH link frames sit differently from the old joint-chain frames the meshes were placed in.
    # Each link needs one fixed correction C so every mesh lands exactly where it did before.
    def _corrections(self):
        R2 = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)
        return [
            SE3.Rx(_H) * SE3.Tz(-D1),                 # link 1 waist
            SE3.Rt(R2, [-L_UPPER, 0, -Y_UP]),         # link 2 shoulder
            SE3(0, -Y_FORE, 0),                       # link 3 elbow
            SE3.Rx(_H) * SE3.Tz(-L_J5),               # link 4 wrist roll
            SE3(0, -J6_OFFSET_Y, 0),                  # link 5 wrist pitch
            SE3(),                                    # link 6 wrist yaw (and the gripper)
        ]

    def link_visuals(self):
        base_plate, parts = LynxmotionSESProET.link_visuals(self)   # same meshes and offsets as before
        C = self._corrections()
        parts = [(i, shape, off if i < 0 else C[i] * off) for i, shape, off in parts]
        return base_plate, parts

    def _frames(self, q):
        """World pose of the base then each link frame (DH frames)."""
        T = self.base
        frames = [T]
        for link, qi in zip(self.links, q):
            T = T * link.A(qi)
            frames.append(T)
        return frames

    def pose_visuals(self, q, parts):
        q = np.clip(q, self.qlim[0], self.qlim[1])
        frames = self._frames(q)
        for link_idk, shape, offset in parts:
            frame = self.base if link_idk < 0 else frames[link_idk + 1]
            shape.T = (frame * offset).A


if __name__ == "__main__":
    import sys, runpy
    sys.modules.setdefault("lynxmotion_ses_pro_model", sys.modules[__name__])
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "demos_and_debug", "lynxmotion_ses-pro_movement.py"), run_name="__main__")