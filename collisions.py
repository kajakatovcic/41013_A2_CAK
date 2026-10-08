# Collision geometry methods for the system workcell.
#
# Two types of collision geometry are used together to catch all collisions:
#
# 1. LINE-PLANE INTERSECTION (taken from lab 5):
#    For each link segment (the straight line between consecutive joint frames)
#    and each triangle of an obstacle mesh, find where the segment crosses the
#    triangle's plane with line_plane_intersection, then test whether that point
#    lies inside the triangle (barycentric test).
#      Pros: A segment that passes through a face is always caught, however
#        thin the face is. The test is exact (not sampled), and the segment has 
#        no thickness so it
#      Cons: The segment has NO thickness. It is the centre-line of a link so a
#        bulky part can touch a face while its centre-line misses it.
#    Example use: Arm links - they are long and thin and their centre-lines are a
#    good approximation of their volume. The links are also well clear of the props
#    so the thickness of the links is not needed. The line test is cheap and exact, 
#    so it is used for every link segment and every obstacle face.
#
# 2. ELLIPSOID vs POINT CLOUD (taken from lab 6):)
#    Each obstacle also carries a point cloud on its surface (built from one
#    meshgrid per face). A body is wrapped in an ellipsoid. The points are moved 
#    into the ellipsoid's frame and any point with algebraic distance < 1 is inside 
#    it (get_algebraic_dist).
#      Pros: Has volume - it models how wide and tall a part is unlike the line test.
#            The ellipsoid is a simple shape that is cheap to test against many points.
#      Cons: Sampled - a face thinner than the point spacing, or an ellipsoid smaller
#        than the gap between points, can be missed. An ellipsoid also only
#        approximates a box or cylinder (does not fill the corners).
#    Example use: MiniCobo gripper and the held cup. These are the bulky parts that go
#    into tight spaces (the 0.20m wide machine bay past the dispenser), where
#    thickness decides if there is a collision. As the cup is only 0.09m wide, its centre
#    line alone is not enough to guarantee it clears a wall. The ellipsoid test adds the 
#    thickness of the gripper and cup to the line test, so a collision is caught even if
#    the centre-line misses the obstacle. 
#
# Together: the line test guarantees no face is passed through (no sampling
# gaps), and the ellipsoids add the thickness of the parts that need it. The arm
# links keep the line test only. They stay clear of the props, and adding ellisoids
# for all seven links would increase the cost of every check. 
#
# first_collision() follows Lab 5's "is_collision()": It uses the same loop over
# trajectory rows, link segments and faces, and stops at the first hit.The planner 
# only needs to know if and what a path hits so not every hit needs to be collected.
# Additions:
#   * obstacles carry a name so that the GUI can report what was hit.
#   * obstacles can be rotated as the RectangularPrism is built about the origin
#     and its vertices and face normals are moved by the obstacle's SE3 (the
#     same way Lab 6 moves its point cloud with a transform)
#   * each obstacle's triangle list and point cloud are built once when it is
#     created or moved instead of inside the loop
#   * skip_links ignores frames that sit inside the robot's own mounting;
#   * tool_frames adds additional segments for objects attached to the flange
#     (e.g. gripper and cup) that aren't part of the robot's link chain
#   * tool_ellipsoids adds the Lab 6 ellipsoid test for additional obkects
#     attached to the flange 

from itertools import combinations
from math import ceil

import numpy as np
from spatialgeometry import Cuboid
from spatialmath import SE3
from ir_support import RectangularPrism, line_plane_intersection

# Spacing of obstacle surface points for the ellipsoid test. Must be smaller
# than the smallest ellipsoid radius (0.05m) so an ellipsoid cannot sit between
# points. 0.02m keeps at least two points across any radius.
POINT_SPACING = 0.02


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


def get_algebraic_dist(points, center_point, radii):
    """
    Algebraic distance of each point from anellipsoid. A value below 1 means 
    the point is inside the ellipsoid.
    """
    return np.sum(((points - center_point) / radii) ** 2, axis=1)


def points_inside_ellipsoid(points, T_ellipsoid, radii):
    """
    Rather than moving the ellipsoid, move the points into the
    ellipsoid's own frame with the inverse transform, then use the
    algebraic distance with the ellipsoid at the origin.
    Returns the indices of the points inside.
    """
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    updated_points = (np.linalg.inv(T_ellipsoid) @ points_h.T).T[:, :3]
    algebra_dist = get_algebraic_dist(updated_points, [0, 0, 0], radii)
    return np.where(algebra_dist < 1)[0]


def link_poses(robot, q=None) -> np.ndarray:
    """
    The transform of every joint frame, i.e. the start and end of every
    link. Returns a stack of 4x4 matrices.

    Uses fkine_all() to get the tranforms of all joint frames (start and end
    of every link)
    """
    frames = robot.fkine_all() if q is None else robot.fkine_all(q)
    return np.asarray(frames.A) if hasattr(frames, "A") else np.asarray(frames)


def make_obstacle_mesh(lwh, pose):
    """
    Build the (vertices, faces, face_normals) triangle-mesh description of a
    rectangular obstacle (see lab5 RectangularPrism) at any pose.

    RectangularPrism only builds boxes aligned with the world axes. So the prism
    is built about the origin and then moved by 'pose': vertices by the full
    transform, face normals by its rotation only (a normal is a direction, so
    translation does not apply). This is the transform-the-points step of
    Lab 6 Q2.6-2.7 applied to the prism's vertices.

    param lwh: [length(X), width(Y), height(Z)] of the box, in its own frame
    param pose: SE3 of the box centre (or an [x, y, z] centre, no rotation)
    """
    T = pose.A if isinstance(pose, SE3) else SE3(*pose).A
    vertices, faces, face_normals = RectangularPrism(
        lwh[0], lwh[1], lwh[2], center=[0, 0, 0]
    ).get_data()
    vertices_h = np.hstack((np.asarray(vertices), np.ones((len(vertices), 1))))
    vertices = (T @ vertices_h.T).T[:, :3]
    face_normals = (T[:3, :3] @ np.asarray(face_normals).T).T
    return vertices, np.asarray(faces), face_normals


def box_surface_points(lwh, pose, spacing=POINT_SPACING):
    """
    Point cloud over the six faces of a box for the ellipsoid test.

    Lab 6 builds a cube's surface from a meshgrid per face. A box has
    faces of three different sizes, so each pair of faces gets its own
    meshgrid (Lab 6 instead rotates one face, which only works for a cube).
    The points are made about the origin and moved by 'pose'.
    """
    l, w, h = lwh

    def grid(a, b):
        A, B = np.meshgrid(np.linspace(-a / 2, a / 2, max(2, ceil(a / spacing) + 1)),
                           np.linspace(-b / 2, b / 2, max(2, ceil(b / spacing) + 1)))
        return A.reshape(-1), B.reshape(-1)

    faces = []
    Y, Z = grid(w, h)
    for sx in (-1, 1):
        faces.append(np.column_stack((np.full(Y.shape, sx * l / 2), Y, Z)))
    X, Z = grid(l, h)
    for sy in (-1, 1):
        faces.append(np.column_stack((X, np.full(X.shape, sy * w / 2), Z)))
    X, Y = grid(l, w)
    for sz in (-1, 1):
        faces.append(np.column_stack((X, Y, np.full(X.shape, sz * h / 2))))
    points = np.concatenate(faces)

    T = pose.A if isinstance(pose, SE3) else SE3(*pose).A
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    return (T @ points_h.T).T[:, :3]


class Obstacle:
    """
    A named box obstacle, at any position and rotation.

    Carrying the NAME alongside the mesh is what lets the safety system report
    "collision predicted with 'milk pitcher'" instead of just True. Useful
    for the GUI and for the sequencer to know what obstacle was hit.

    Holds both the triangle mesh (line-plane test) and a surface point cloud 
    (ellipsoid test).

    'movable' marks obstacles that can be repositioned at runtime. e.g the cup
    is movable, but the bay and tray are fixed.
    """

    def __init__(self, name, lwh, pose, movable=False, color=None):
        self.name = name
        self.lwh = np.asarray(lwh, dtype=float)
        self.movable = movable
        self.pose = pose if isinstance(pose, SE3) else SE3(*pose)
        self._build()
        self.shape = Cuboid(
            scale=list(self.lwh),
            pose=self.pose,
            color=color if color is not None else (0.85, 0.25, 0.15, 0.9),
        )
        self.active = True

    def _build(self):
        """
        Rebuild the mesh, its triangle list and its point cloud.
        The triangle_list is built once per obstacle instead of inside the collision
        loop. The triangle_list is a list of all combinations of three vertices so
        that the barycentric test can be applied to each triangle. 
        """
        self.mesh = make_obstacle_mesh(self.lwh, self.pose)
        vertices, faces, face_normals = self.mesh
        self._faces = []
        for j, face in enumerate(faces):
            vert_on_plane = vertices[face][0]
            triangle_list = np.array(list(combinations(face, 3)), dtype=int)
            self._faces.append((face_normals[j], vert_on_plane,
                                [vertices[triangle] for triangle in triangle_list]))
        self.points = box_surface_points(self.lwh, self.pose)

    @property # Property so that the GUI can read it without a method call
    def centre(self):
        return self.pose.t

    def move_to(self, pose):
        """Reposition (SE3, or an [x, y, z] centre), rebuilding the geometry."""
        self.pose = pose if isinstance(pose, SE3) else SE3(*pose)
        self._build()
        self.shape.T = self.pose.A

    def add_to_env(self, env):
        env.add(self.shape)


def segment_hit(p_start, p_end, obstacle):
    """
    Lab 5 line-plane test for ONE segment against ONE obstacle. Returns the
    intersection point or None. Used for the light-curtain beams.
    """
    for face_normal, vert_on_plane, triangle_list in obstacle._faces:
        intersect_p, check = line_plane_intersection(face_normal, vert_on_plane,
                                                     p_start, p_end)
        if check == 1:
            for triangle in triangle_list:
                if is_intersection_point_inside_triangle(intersect_p, triangle):
                    return np.asarray(intersect_p)
    return None


def first_collision(robot, q_matrix, obstacles, skip_links=0, tool_frames=None,
                    tool_ellipsoids=None):
    """
    Uses lab 5's is_collision() with the lab 6 ellipsoid test added.
    The named-obstacle version of Lab 5's is_collision(), with the Lab 6
    ellipsoid test added.

    Returns the name of the first obstacle hit along the given trajectory or
    None if the whole path is clear.

    param skip_links: Leading link frames to ignore. This is needed when a robot is
        MOUNTED on something that is also an obstacle, as its base frames sit inside
        that geometry and would read as a permanent hit.
    param tool_frames: optional function flange_T (4x4). Additional frames (4x4)
        for objects attached to the flange (e.g. gripper and cup). Fkine_all() stops at
        the flange so without this the gripper and anything carried are invisible to the 
        line-plane test. 
    param tool_ellipsoids: optional function flange_T (4x4). Additional list of
        (T_ellipsoid 4x4, radii) for the Lab 6 test to give additional objects attached to
        the flange their real width and height.
    """
    active = [o for o in obstacles if o.active]
    for q in q_matrix:
        # Get the transform of every joint (i.e. start and end of every link)
        tr = link_poses(robot, q)
        T_flange = tr[-1]
        if tool_frames is not None:
            tr = np.concatenate([tr, np.asarray(tool_frames(T_flange))], axis=0)

        # go through each link and also each triangle face of each obstacle (see lab5)
        for i in range(skip_links, np.shape(tr)[0] - 1):
            for obs in active:
                for face_normal, vert_on_plane, triangle_list in obs._faces:
                    intersect_p, check = line_plane_intersection(face_normal,
                                                                 vert_on_plane,
                                                                 tr[i][:3, 3],
                                                                 tr[i + 1][:3, 3])
                    if check == 1:
                        for triangle in triangle_list:
                            if is_intersection_point_inside_triangle(intersect_p, triangle):
                                return obs.name

        # obstacle surface points inside the gripper / cup ellipsoids (see lab6)
        if tool_ellipsoids is not None:
            for T_ellipsoid, radii in tool_ellipsoids(T_flange):
                for obs in active:
                    inside = points_inside_ellipsoid(obs.points, T_ellipsoid, radii)
                    if inside.size > 0:
                        return obs.name
    return None


def is_collision(robot, q_matrix, obstacles, skip_links=0, tool_frames=None,
                 tool_ellipsoids=None):
    """Boolean form of first_collision (Lab 5's return value)."""
    return first_collision(robot, q_matrix, obstacles, skip_links, tool_frames,
                           tool_ellipsoids) is not None
