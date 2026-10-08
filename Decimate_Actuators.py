import os
import trimesh
 
TARGET_FACES = 15000
HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(HERE, "Lynxmotion_Meshes")
 
for name in ["lss_m1", "lss_s1", "lss_l1"]:
    src = os.path.join(MESH_DIR, name + ".stl")
    dst = os.path.join(MESH_DIR, name + "_lo.stl")
    if not os.path.isfile(src):
        print(name, ": file not found ->", src)
        continue
    m = trimesh.load(src, force="mesh")
    before = len(m.faces)
    if before > TARGET_FACES:
        m = m.simplify_quadric_decimation(face_count=TARGET_FACES)
    m.export(dst)
    print("%s: %d -> %d faces, extents %s, saved %s"
          % (name, before, len(m.faces), m.extents.round(4).tolist(), os.path.basename(dst)))