"""Pack the flybody fruit fly (Vaxenburg et al. 2025, Apache-2.0) for the web viewer.

MuJoCo compiles the MJCF (so mesh recentering and frames are exactly MuJoCo's); each visual mesh
is moved into its body's frame, welded, decimated (pyfqmr) and given smooth normals. Output:
fly.gz.b64.txt (meshes) and fly_skeleton.json (bodies, hinge joints with axes, anchors, ranges).

    python scripts/setup/pack_flybody.py external/flybody/flybody/fruitfly/assets/fruitfly.xml viz/cns
"""

import base64
import gzip
import json
import struct
import sys
from pathlib import Path

import mujoco
import numpy as np
import pyfqmr


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    xml, out = sys.argv[1], Path(sys.argv[2])
    keep = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
    m = mujoco.MjModel.from_xml_path(xml)
    name = lambda kind, i: mujoco.mj_id2name(m, kind, i) or f"{i}"
    bodies = [dict(name=name(mujoco.mjtObj.mjOBJ_BODY, b), parent=int(m.body_parentid[b]),
                   pos=m.body_pos[b].tolist(), quat=m.body_quat[b].tolist()) for b in range(m.nbody)]
    joints = [dict(name=name(mujoco.mjtObj.mjOBJ_JOINT, j), body=int(m.jnt_bodyid[j]), type=int(m.jnt_type[j]),
                   axis=m.jnt_axis[j].tolist(), pos=m.jnt_pos[j].tolist(), range=m.jnt_range[j].tolist())
              for j in range(m.njnt)]
    parts, total_in, total_out = [], 0, 0
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g] != 1:
            continue
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        v = m.mesh_vert[va:va + vn].astype(np.float64)
        f = m.mesh_face[fa:fa + fn].astype(np.int64)
        v = v @ quat_mat(m.geom_quat[g]).T + m.geom_pos[g]  # geom frame -> body frame
        # weld duplicate corners, then decimate
        key = np.round(v / 1e-6).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        vw = np.zeros((inv.max() + 1, 3)); vw[inv] = v
        fw = inv[f]
        fw = fw[(fw[:, 0] != fw[:, 1]) & (fw[:, 1] != fw[:, 2]) & (fw[:, 0] != fw[:, 2])]
        target = max(120, int(len(fw) * keep))
        s = pyfqmr.Simplify()
        s.setMesh(vw, fw)
        s.simplify_mesh(target_count=target, aggressiveness=6, preserve_border=True, verbose=False)
        vs, fs, _ = s.getMesh()
        nrm = np.zeros_like(vs)
        tri = vs[fs]
        fnrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for c in range(3):
            np.add.at(nrm, fs[:, c], fnrm)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        rgba = m.mat_rgba[m.geom_matid[g]] if m.geom_matid[g] >= 0 else m.geom_rgba[g]
        mat = name(mujoco.mjtObj.mjOBJ_MATERIAL, m.geom_matid[g]) if m.geom_matid[g] >= 0 else ""
        parts.append(dict(body=int(m.geom_bodyid[g]), rgba=rgba.tolist(), mat=mat, v=vs, n=nrm, f=fs))
        total_in += fn; total_out += len(fs)
    allv = np.concatenate([p["v"] for p in parts])
    lo, hi = allv.min(0), allv.max(0)
    span = float(np.abs(allv).max())
    blob = [struct.pack("<If", len(parts), span)]
    meta = []
    for p in parts:
        q = np.clip(np.round(p["v"] / span * 32767), -32767, 32767).astype(np.int16)
        nq = np.clip(np.round(p["n"] * 127), -127, 127).astype(np.int8)
        idx = p["f"].astype(np.uint32)
        blob.append(struct.pack("<III", len(q), len(idx), p["body"]))
        blob.append(q.tobytes()); blob.append(np.pad(nq, ((0, 0), (0, 1))).tobytes()); blob.append(idx.tobytes())
        meta.append(dict(body=p["body"], rgba=[round(x, 3) for x in p["rgba"]], mat=p["mat"], nv=len(q), nf=len(idx)))
    payload = b"".join(blob)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fly.gz.b64.txt").write_text(base64.b64encode(gzip.compress(payload, 9)).decode())
    (out / "fly_skeleton.json").write_text(json.dumps(dict(bodies=bodies, joints=joints, parts=meta, span=span,
                                                           units="cm", source="flybody (Vaxenburg et al. 2025), Apache-2.0")))
    print(f"{len(parts)} visual meshes, faces {total_in} -> {total_out}; payload {len(payload) / 1e6:.2f} MB, "
          f"gz+b64 {(out / 'fly.gz.b64.txt').stat().st_size / 1e6:.2f} MB; bbox {lo.round(3)} .. {hi.round(3)} cm")
    print("materials:", sorted(set(p["mat"] for p in meta)))


if __name__ == "__main__":
    main()
