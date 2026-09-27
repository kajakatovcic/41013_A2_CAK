# Collision geometry methods for the system workcell.
# METHOD - derived from lab 5: for each link SEGMENT (the straight line between
# consecutive joint frames) and each triangle of an obstacle mesh, find where the
# segment crosses the triangle's plane with line_plane_intersection, then test
# whether that crossing point actually lies inside the triangle (barycentric test).
# This catches a link sweeping THROUGH an obstacle, which a point-only test on the
# joint frames cannot see.

import numpy as np
from spatialgeometry import Cuboid
from spatialmath import SE3
from ir_support import RectangularPrism, line_plane_intersection
def is_intersection_point_inside_triangle(intersect_p, triangle_verts) -> bool:
    """
    Barycentric test (Lab 5) for whether a plane-intersection point lies inside a
    triangle, as opposed to merely somewhere on the triangle's infinite plane.
    """
    # Pick one corner as the origin and describe every point in the plane using
    # the two edges leaving that corner.
    u = triangle_verts[1, :] - triangle_verts[0, :]
    v = triangle_verts[2, :] - triangle_verts[0, :]

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)

    w = intersect_p - triangle_verts[0, :]
    wu = np.dot(w, u)
    wv = np.dot(w, v)

    D = uv * uv - uu * vv
    if abs(D) < 1e-12: # degenerate triangle, cannot be hit
        return False

    # Get and test the parametric coordinates (s and t)
    s = (uv * wv - vv * wu) / D
    if s < 0.0 or s > 1.0: # intersect_p is outside the triangle
        return False

    t = (uv * wu - uu * wv) / D
    if t < 0.0 or (s + t) > 1.0: # intersect_p is outside the triangle
        return False

    return True # intersect_p is inside the triangle


def link_poses(robot, q=None) -> np.ndarray:
    """
    The transform of every joint frame, i.e. the start and end of every
    link. Returns a stack of 4x4 matrices.

    Works for a DHRobot, a DHRobot3D/UTSMeshRobot (the JAKA MiniCobo) and an
    ERobot (the Lynxmotion), as all three implement fkine_all().
    """
    frames = robot.fkine_all() if q is None else robot.fkine_all(q)
    return np.asarray(frames.A) if hasattr(frames, "A") else np.asarray(frames)

def make_obstacle_mesh(lwh, centre):
    """
    Build the (vertices, faces, face_normals) triangle-mesh description of a
    rectangular obstacle, (see Lab 5 q2.3, RectangularPrism)

    param lwh: [length(X), width(Y), height(Z)] of the box
    param centre: [x, y, z] centre of the box
    """
    vertices, faces, face_normals = RectangularPrism(
        lwh[0], lwh[1], lwh[2], center=centre
    ).get_data()
    return vertices, faces, face_normals