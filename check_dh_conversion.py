"""
Checks the DH Lynxmotion against the old joint-chain model (no Swift needed).

    python check_dh_conversion.py

Compares, at 200 random joint states, the end-effector pose and the world pose of every mesh
and gripper part. Everything should be ~1e-12 or smaller. Once it passes, delete
LynxmotionSESProET from lynxmotion_ses_pro_model.py.
"""
import numpy as np
from lynxmotion_ses_pro_model import LynxmotionSESPro, LynxmotionSESProET

rng = np.random.default_rng(0)
new, ref = LynxmotionSESPro(), LynxmotionSESProET()
_, parts_new = new.link_visuals()
_, parts_ref = ref.link_visuals()
assert len(parts_new) == len(parts_ref)

lo, hi = np.asarray(new.qlim[0]), np.asarray(new.qlim[1])
worst_fk = worst_part = 0.0
for _ in range(200):
    q = rng.uniform(lo, hi)
    new.pose_visuals(q, parts_new)
    ref.pose_visuals(q, parts_ref)
    worst_fk = max(worst_fk, float(np.abs(new.fkine(q).A - ref.fkine(q).A).max()))
    for (_, a, _), (_, b, _) in zip(parts_new, parts_ref):
        worst_part = max(worst_part, float(np.abs(np.asarray(a.T) - np.asarray(b.T)).max()))

print(f"end-effector pose, worst difference : {worst_fk:.2e}")
print(f"mesh/gripper part poses, worst diff : {worst_part:.2e}")
print("PASS" if worst_fk < 1e-9 and worst_part < 1e-9 else "FAIL: do not use yet, paste this output to me")