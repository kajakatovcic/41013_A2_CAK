"""
Lynxmotion SES-PRO 900mm 6DoF arm - the MODEL (geometry, kinematics, meshes).

Everything about what the robot IS lives here: measuring the STL meshes, the link offsets, the
actuator/tube placement, the joint limits and the ERobot class with its visuals.
Everything about what the robot DOES (poses, waypoints, animation) lives in "Lynxmotion SES-PRO.py".

Keep this file in the same folder as "Lynxmotion SES-PRO.py" (the repo folder 41013_A2_CAK).
Run it directly (python3 lynxmotion_model.py) for a quick no-Swift sanity check.
"""
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
    return 4 * np.pi * h.volume / h.area ** 2     


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
                hub_sign=hub_sign, hub_counts=(n_lo, n_hi), verts=V)


MEAS = {k: measure(k) for k in FILES}

# ------------------------------------------------------------
# SLIM TUBES: the tube STL is fatter than the actuator housings' plain body, so its end pokes out of
# the elbow (J3) and the pitch actuator (J5). The real tube is 70 mm OD (SKU CF-70x67-280.5), so
# make a slimmer copy that is always narrower than the narrowest housing it plugs into.
# ------------------------------------------------------------
TUBE_OD_MAX = 0.070        # real tube outer diameter (m)
TUBE_FIT = 0.92            # tube radius as a fraction of the narrowest housing BODY it meets


def _body_radius(kind):
    """Radius of the plain housing BODY (middle of the actuator), not the wider hub flanges.
    Uses points sampled over the surface (the decimated STLs have no vertices mid-way along the side)."""
    M = MEAS[kind]; ia = M["ia"]; c = M["centre"]
    mesh = trimesh.load(os.path.join(MESH_DIR, FILES[kind]))
    try:
        S, _ = trimesh.sample.sample_surface(mesh, 60000, seed=0)
    except TypeError:                                   # older trimesh without 'seed'
        np.random.seed(0)
        S, _ = trimesh.sample.sample_surface(mesh, 60000)
    p, q = [i for i in range(3) if i != ia]
    mid = np.abs(S[:, ia] - c[ia]) < 0.15 * M["length"]
    r = np.hypot(S[mid, p] - c[p], S[mid, q] - c[q])
    r = r[r > 0.5 * M["radius"]]
    hist, edges = np.histogram(r, bins=80)
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


_R_BODY = {k: _body_radius(k) for k in ("m1", "s1", "l1")}
_TUBE_SCALE = min(1.0,
                  TUBE_FIT * min(_R_BODY.values()) / MEAS["tube"]["radius"],
                  0.5 * TUBE_OD_MAX / MEAS["tube"]["radius"])


def _slim_tube_path():
    """Writes a copy of the tube STL narrowed about its own axis (length unchanged)."""
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
print(f"tube radius {MEAS['tube']['radius']*1000:.1f} mm -> {MEAS['tube']['radius']*_TUBE_SCALE*1000:.1f} mm "
      f"(housing bodies: " + ", ".join(f"{k} {v*1000:.1f}" for k, v in _R_BODY.items()) + " mm)")

X_, Y_, Z_ = (1, 0, 0), (0, 1, 0), (0, 0, 1)


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


def _pts(kind, pose):
    return MEAS[kind]["verts"] @ pose.R.T + pose.t


def _surface(V, axis, sign, mask, bin_=0.0005):
    if int(mask.sum()) < 8:
        return None
    v = sign * V[mask, axis]
    v = v[v >= np.percentile(v, 60)]
    bins = np.floor((v - v.min()) / bin_).astype(int)
    counts = np.bincount(bins)
    best = len(counts) - 1 - int(np.argmax(counts[::-1]))    
    return sign * float(v[bins == best].mean())

def _housing_offset(V, c0, axis, length, R):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    u = np.cross(a, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(a, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u); v = np.cross(a, u)
    d = V - np.asarray(c0, float)
    t, x, y = d @ a, d @ u, d @ v
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

# ------------------------------------------------------------
# DIMENSIONS
# ------------------------------------------------------------
TUBE_LEN = MEAS["tube"]["length"]     
TUBE_R = MEAS["tube"]["radius"] * _TUBE_SCALE
EMB = 0.02                      # (legacy) tube end sunk into the actuator housing; no longer used for J5
OV = 0.02                       # how far a tube's side intrudes past an actuator's END FACE (elbow)
SH_SINK = 0.002                 # how far tube 1's flat ends sink into the shoulder / elbow (metres).
                                # 0 = just touching, bigger = deeper. Never negative (that opens a gap).
SHOULDER_STRIP_TO_TUBE = True   # True: the shoulder's holes panel faces the tube, so the tube
                                # end sits on that panel. False: it sits on the plain curved side.
# Elbow actuator orientation, built from two quarter turns applied to the "facing the camera" pose
# (cable/logo end pointing at the camera, connector panel down). Seen from the usual camera view:
ELBOW_SPIN_DEG = -90            # spin about the vertical: -90 turns it to the LEFT (+90 = right), so the
                                # cable/logo end faces left
ELBOW_ROLL_DEG = 0           # then roll about its own (now left-right) axis: 90 rolled the top AWAY
                                # from the camera; 270 (= -90) is that plus another 180 deg.
                                # Adding or subtracting 180 gives the same result either way.
# Wrist-roll actuator (J4) orientation, from the same camera view: starting pose is axis up with the
# connector panel facing the camera; spin it about the vertical, then flip it end-for-end about the
# left-right axis (toward/away from the camera gives the same result for 180 deg).
J4_SPIN_DEG = -90               # -90 = spin to the LEFT (+90 = right)
J4_FLIP_DEG = 180               # then 180 about the left-right axis (0 = leave it)
# Wrist-pitch actuator (J5) orientation, seen end-on from the camera (looking along +Y at its end face):
# rotate it ANTICLOCKWISE 45 deg about its own axis, then flip it end-for-end so the other side faces us.
J5_SPIN_DEG = 0              # about its own (Y) axis; -45 = anticlockwise as seen from the camera at -Y
J5_FLIP_DEG = 180              # then turn it around (0 = leave it)
J5_TURN_DEG = 90               # final turn about the same axis AFTER the flip, seen from the camera:
                               # +90 = clockwise, -90 = anticlockwise, 0 = none
J5_FLIP_ABOUT = "z"            # flip about the tube direction ("z") or about the sideways axis ("x")
# Wrist-yaw actuator (J6): seen from the camera looking down its axis (gripper end towards us),
# spin it ANTICLOCKWISE 90 deg about its own axis.
J6_SPIN_DEG = 90               # +90 = anticlockwise as seen from the camera, -90 = clockwise
J6_SINK = 0.002                # how far J6's housing sinks into J5's end face (0 = just touching)
J6_NUDGE_Y = 0.0               # extra sideways nudge of J6 along J5's axis (metres); + = away from J5
J4_SINK = 0.002                # how far tube 2 sinks into the roll actuator's top face (0 = just touching)
# Tube ends are buried in the actuator housing. Elbow and J5 have the same radius as the tube, so the end only
# disappears completely when it reaches the actuator's AXIS. These say how far BELOW that axis the tube's end
# stops (metres): 0 = exactly on the axis, bigger = shallower. Keep it small (0 to 0.006). If a sliver of the
# tube end shows at the sides raise it a little; if the tube pokes out of the far side of the actuator, lower it.
TUBE1_END_TO_AXIS = 0.002      # tube 1's top end -> elbow axis
TUBE2_END_TO_AXIS = 0.002      # tube 2's top end -> J5 axis
ELBOW_CENTRE_ON_BODY = True    # True: centre tube 1's top on the elbow's real housing (ignores hub lip/studs)
ELBOW_FLIP = False             # True swaps the elbow's two end faces (hub end <-> plain end)
J4_GAP = 0.002                 # minimum clearance between the roll actuator (J4) and tube 1

LEN_M1, LEN_S1, LEN_L1 = (MEAS[k]["length"] for k in ("m1", "s1", "l1"))
R_M1, R_S1, R_L1 = (MEAS[k]["radius"] for k in ("m1", "s1", "l1"))
CLAMP = 0.015              # overlap used where an actuator sits on another actuator

# J1/J2: the shoulder cylinder sticks out sideways (+Y) from the waist's connector panel
# (the side with the holes), with its axis through the middle of that panel.
# The shoulder is turned so its HUB end (the face with the holes, found automatically
# in measure()) points along +Y, where tube 1 is clamped on.
WAIST_DX = 0.0             # nudge the waist actuator sideways (metres) if you want to
WAIST_DY = 0.0
SHOULDER_FLIP = False      # True swaps the shoulder's two end faces (hub end <-> plain end)
SEAT = 0.002               # how far the shoulder's end face sinks into the waist's panel (0 = touching)
_Mw = MEAS["m1"]
WAIST_PANEL_Z = LEN_M1 / 2 + (_Mw["bump_mid"] - _Mw["centre"][_Mw["ia"]])
D1 = WAIST_PANEL_Z                        # shoulder axis height above base frame

# --- J1 -> J2: snap the shoulder's end face onto the waist's real connector panel ---------
_Pw = place("m1", Z_, (0, 1, 0), (WAIST_DX, WAIST_DY, LEN_M1 / 2))     # waist pose (base frame)
_Vw = _pts("m1", _Pw)
_mw = ((np.abs(_Vw[:, 0] - WAIST_DX) < 0.5 * R_M1) & (np.abs(_Vw[:, 2] - D1) < 0.5 * R_M1)
       & (_Vw[:, 1] > WAIST_DY))
_panel_y = _surface(_Vw, 1, +1, _mw)                                   # panel plane (y)
WAIST_PANEL_Y = _panel_y if _panel_y is not None else WAIST_DY + R_M1 + _Mw["bump_h"]
SH_FACE_Y = WAIST_PANEL_Y - SEAT          # shoulder's -Y end face
SH_YC = SH_FACE_Y + LEN_M1 / 2            # shoulder body centre along its axis
SH_END = SH_YC + LEN_M1 / 2               # +Y end face of the shoulder actuator
Y_UP = SH_YC                              # tube 1 lies in the MIDDLE plane of the shoulder actuator

# --- J2 -> tube 1: snap tube 1's flat end onto the shoulder's real surface ------------------
SH_AXIS = (0, MEAS["m1"]["hub_sign"] * (-1 if SHOULDER_FLIP else 1), 0)
SH_STRIP = Z_ if SHOULDER_STRIP_TO_TUBE else (-1, 0, 0)    # shoulder's holes panel faces the tube
_Vs = _pts("m1", place("m1", SH_AXIS, SH_STRIP, (0, SH_YC, 0)))        # shoulder pose (shoulder frame)
_ms = ((np.abs(_Vs[:, 0]) < 0.6 * TUBE_R) & (np.abs(_Vs[:, 1] - SH_YC) < 0.6 * TUBE_R)
       & (_Vs[:, 2] > 0))
_top = _surface(_Vs, 2, +1, _ms)
SH_TUBE_TOP = _top if _top is not None else R_M1 + (_Mw["bump_h"] if SHOULDER_STRIP_TO_TUBE else 0.0)
SH_TUBE_Z0 = SH_TUBE_TOP - SH_SINK        # tube 1 starts here (z from the shoulder axis)

# J2->J3: tube 1 straight up; elbow origin sits on the upper-arm plane
# The elbow actuator lies along Y (parallel to the shoulder), centred on tube 1's plane, with
# its cable/logo end facing left. Start from "cable end at the camera (-X), connector panel
# down", spin it about Z (ELBOW_SPIN_DEG), then roll it about Y (ELBOW_ROLL_DEG).
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

ELB_BODY_YC = 0.0                         # elbow body centre (y, elbow frame)

# --- tube 1 -> J3: centre the elbow's HOUSING on tube 1, then snap its underside onto the tube end
_V0 = _pts("s1", place("s1", EL_AXIS, EL_STRIP, (0, 0, 0)))
_d_el = (_housing_offset(_V0, np.zeros(3), EL_AXIS, LEN_S1, R_S1)
         if ELBOW_CENTRE_ON_BODY else np.zeros(3))
EL_POS = tuple(-_d_el)                    # elbow mesh position: housing centre lands on tube 1's axis
_Ve = _pts("s1", place("s1", EL_AXIS, EL_STRIP, EL_POS))               # elbow pose (elbow frame)
_me = ((np.abs(_Ve[:, 0]) < 0.6 * TUBE_R) & (np.abs(_Ve[:, 1]) < 0.6 * TUBE_R) & (_Ve[:, 2] < 0))
_bot = _surface(_Ve, 2, -1, _me)                                       # negative number
EL_BOTTOM = _bot if _bot is not None else -(R_S1 + (_Ms["bump_h"] if EL_STRIP[2] < -0.5 else 0.0))
L_UPPER = SH_TUBE_Z0 + TUBE_LEN + TUBE1_END_TO_AXIS    # elbow AXIS height above the shoulder axis (tube 1's end is buried up to it)

# J4 (roll actuator, axis Z) is bolted to the elbow's -Y end face (J4's side meets that face,
# sunk 3 mm), and kept clear of tube 1 so it cannot clip it when the elbow turns.
Y_FORE = -max(LEN_S1 / 2 + R_S1 - 0.003 - EL_POS[1],  # touching the elbow's -Y end face
              TUBE_R + R_S1 + J4_GAP)              # ...but never closer than tube 1 allows
TUBE_CLEARANCE = abs(Y_FORE) - R_S1 - TUBE_R   # J4 vs tube 1 clearance (must be > 0)

# J4 -> J5: tube 2 grows out of J4's top face; J5 (pitch, axis Y) sits on the tube's end
_Vj4 = _pts("s1", place("s1", J4_AXIS, J4_STRIP))                      # J4 pose (J4 frame)
_m4 = ((np.abs(_Vj4[:, 0]) < 0.9 * TUBE_R) & (np.abs(_Vj4[:, 1]) < 0.9 * TUBE_R) & (_Vj4[:, 2] > 0))
_j4t = _surface(_Vj4, 2, +1, _m4)                                      # J4's real top face
J4_TOP = _j4t if _j4t is not None else LEN_S1 / 2
TUBE2_Z0 = J4_TOP - J4_SINK               # tube 2 starts at J4's top face, sunk in by J4_SINK only

# tube 2's far end meets J5's real underside (J5 is a cylinder lying on its side), sunk in by J5_SINK
_V5 = _pts("l1", place("l1", J5_AXIS, J5_STRIP))                       # J5 pose (J5 frame)
_m5 = ((np.abs(_V5[:, 0]) < 0.9 * TUBE_R) & (np.abs(_V5[:, 1]) < 0.9 * TUBE_R) & (_V5[:, 2] < 0))
_j5b = _surface(_V5, 2, -1, _m5)                                       # negative number
J5_BOTTOM = _j5b if _j5b is not None else -R_L1
L_J5 = TUBE2_Z0 + TUBE_LEN + TUBE2_END_TO_AXIS            # J5 AXIS height (tube 2's end is buried up to it)

# J5 -> J6:
# J6 (axis Z) stands on J5's +Y end face. Its centre is placed so its curved side (plus its
# connector panel, if that faces J5) just touches the REAL end face of J5, sunk in by J6_SINK.
# Because J6 turns about its own axis, nothing of J6 can swing into J5 at any joint angle.
_Vj5 = _pts("l1", place("l1", J5_AXIS, J5_STRIP))                      # J5 pose (J5 frame)
_mj = ((np.abs(_Vj5[:, 0]) < 0.9 * R_L1) & (np.abs(_Vj5[:, 2]) < 0.9 * R_L1) & (_Vj5[:, 1] > 0))
_j5e = _surface(_Vj5, 1, +1, _mj)
J5_END_Y = _j5e if _j5e is not None else LEN_L1 / 2
J6_TOWARD_J5 = J6_STRIP[1] < -0.5          # J6's connector panel faces J5
_Vj6 = _pts("l1", place("l1", Z_, J6_STRIP))                           # J6 pose (J6 frame)
_mj6 = ((np.abs(_Vj6[:, 0]) < 0.9 * R_L1) & (np.abs(_Vj6[:, 2]) < 0.4 * LEN_L1) & (_Vj6[:, 1] < 0))
_j6f = _surface(_Vj6, 1, -1, _mj6)                                     # J6's real face towards J5 (y < 0)
J6_FACE_Y = _j6f if _j6f is not None else -(R_L1 + (MEAS["l1"]["bump_h"] if J6_TOWARD_J5 else 0.0))
J6_SIDE_X = 0
J6_OFFSET = J6_SIDE_X * (R_L1 + R_L1 - CLAMP)
J6_OFFSET_Y = J5_END_Y - J6_SINK - J6_FACE_Y + J6_NUDGE_Y              # J6's face meets J5's end face

# J6 -> gripper: gripper hangs below the bottom of J6
TOOL_Z = -(LEN_L1 / 2 + 0.085)

# Joint limits (rad), from the Lynxmotion SES-PRO 900mm 6DoF spec: J1 +-360 deg (cable slack),
# J2..J5 +-180 deg, J6 +-180 deg with a cabled gripper (infinite without one).
QLIMS = [[-2 * np.pi, 2 * np.pi],  # J1 waist
         [-np.pi, np.pi],          # J2 shoulder
         [-np.pi, np.pi],          # J3 elbow
         [-np.pi, np.pi],          # J4 wrist roll
         [-np.pi, np.pi],          # J5 wrist pitch
         [-np.pi, np.pi]]          # J6 wrist yaw

class LynxmotionSESPro(ERobot):

    def __init__(self):

        l1 = Link(ET.Rz(qlim=QLIMS[0]), name="waist")
        l2 = Link(ET.tz(D1) * ET.Ry(qlim=QLIMS[1]), name="shoulder", parent=l1)
        l3 = Link(ET.ty(Y_UP) * ET.tz(L_UPPER) * ET.Ry(qlim=QLIMS[2]),
                  name="elbow", parent=l2)
        l4 = Link(ET.ty(Y_FORE) * ET.Rz(qlim=QLIMS[3]),
                  name="wrist_roll", parent=l3)
        l5 = Link(ET.tz(L_J5) * ET.Ry(qlim=QLIMS[4]), name="wrist_pitch", parent=l4)
        l6 = Link(
            ET.tx(J6_OFFSET) * ET.ty(J6_OFFSET_Y) * ET.Rz(qlim=QLIMS[5]),
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
        # pose that matches the product photo: upper arm leaning back, forearm
        # leaning forward, wrist pitched back so the end block is vertical
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
            return Mesh(filename=SLIM_TUBE_PATH, scale=[1, 1, 1], color=TUBE_COL)

        S = (-1, 0, 0)                       # connector strip faces backwards
        # shoulder orientation (SH_AXIS / SH_STRIP) and elbow orientation (EL_AXIS / EL_STRIP)
        # are computed once at module level
        tube_z = SH_TUBE_Z0 + TUBE_LEN / 2   # tube 1 mesh is centred, axis along Z
        tube2_z = TUBE2_Z0 + TUBE_LEN / 2   # tube 2 starts at J4's top face

        parts = [
            # (frame index, shape, pose in that frame)   -1 = robot base frame
            # J1 waist: vertical, sitting on the bracket. It rides on the WAIST joint (frame 0), so
            # its connector panel (the side with the holes, facing +Y towards the shoulder) keeps
            # facing the shoulder when the waist turns.
            (0, actuator("m1"), place("m1", Z_, (0, 1, 0), (WAIST_DX, WAIST_DY, LEN_M1 / 2))),

            # J2 shoulder: axis Y, HUB end (face with the holes) pointing +Y where tube 1 sits.
            # hub_sign says which end of the mesh is the hub, so the axis is chosen to put it on +Y.
            (1, actuator("m1"), place("m1", SH_AXIS, SH_STRIP, (0, SH_YC, 0))),
            (1, tube(), place("tube", Z_, S, (0, Y_UP, tube_z))),

            # J3 elbow: axis Y. Its BODY is clamped to tube 1, so it stays in the upper-arm frame
            # (frame 1) at the elbow's rest position; the forearm turns relative to it.
            (1, actuator("s1"), SE3(0, Y_UP, L_UPPER) * place("s1", EL_AXIS, EL_STRIP, EL_POS)),

            # J4 wrist roll: axis Z. Its body is bolted to the elbow's output end face, so it rides on
            # the elbow joint (frame 2) at y = Y_FORE; only tube 2 (frame 3) rolls relative to it.
            (2, actuator("s1"), SE3(0, Y_FORE, 0) * place("s1", J4_AXIS, J4_STRIP)),
            # tube 2 grows out of J4's top face (rolls with J4)
            (3, tube(), place("tube", Z_, S, (0, 0, tube2_z))),

            # J5 wrist pitch: axis Y. Its body sits on tube 2, so it rides on the forearm (frame 3)
            # at the pitch height; only J6 (and the gripper) turn relative to it.
            (3, actuator("l1"), SE3(0, 0, L_J5) * place("l1", J5_AXIS, J5_STRIP)),

            # J6 wrist yaw: axis Z. Its body is bolted to J5's output end face, so it rides on the
            # pitch joint (frame 4), offset along J5's axis so J6's side meets J5's end face; only the
            # gripper (frame 5) turns about J6's axis.
            (4, actuator("l1"), SE3(J6_OFFSET, J6_OFFSET_Y, 0) * place("l1", Z_, J6_STRIP)),
        ]

        if WITH_GRIPPER:
            z0 = -LEN_L1 / 2

            parts += [
                # Gripper mounting stem below J6
                (5, Cylinder(radius=0.02, length=0.06, color=GRIP),
                SE3(0, 0, z0 - 0.03)),

                # Left finger
                (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
                SE3(0.02, 0, z0 - 0.085)),

                # Right finger
                (5, Cuboid(scale=[0.012, 0.015, 0.05], color=GRIP),
                SE3(-0.02, 0, z0 - 0.085)),
            ]
        return base_plate, parts

    def pose_visuals(self, q, parts):
        q = np.clip(q, self.qlim[0], self.qlim[1])      # stay inside the joint limits
        link_frames = self.fkine_all(q)
        for link_idk, shape, offset in parts:
            if link_idk < 0:
                frame = self.base
            else:
                frame = link_frames[link_idk + 1]
            shape.T = (frame * offset).A


if __name__ == "__main__":
    # quick no-Swift sanity check of the model
    robot = LynxmotionSESPro()
    print(robot)
    print("tool tip at qz:", np.round(robot.fkine(robot.qz).t, 3))
    print("tool tip at qr:", np.round(robot.fkine(robot.qr).t, 3))