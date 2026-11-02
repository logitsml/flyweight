"""FlyWire v783 annotations (Schlegel et al. 2024) aligned to the model's neuron order."""

from functools import lru_cache

import numpy as np
import pandas as pd

from .connectome import ROOT, VERSIONS

ANNOT = ROOT / "external" / "flywire_annotations_v783.tsv"


@lru_cache(maxsize=1)
def load(version="783"):
    comp = pd.read_csv(VERSIONS[version][0], index_col=0)
    a = pd.read_csv(ANNOT, sep="\t", low_memory=False).drop_duplicates("root_id").set_index("root_id")
    a = a.reindex(comp.index)
    # FlyWire coordinates are in 4 x 4 x 40 nm voxels; prefer the soma, else a point on the arbor
    xyz = np.stack([a.soma_x.fillna(a.pos_x), a.soma_y.fillna(a.pos_y), a.soma_z.fillna(a.pos_z)], 1)
    xyz = xyz * np.array([4.0, 4.0, 40.0])
    name = a.cell_type.fillna(a.hemibrain_type).fillna("").astype(str).values
    return dict(
        flyids=comp.index.values.astype(np.int64),
        xyz=xyz.astype(np.float32),
        super_class=a.super_class.fillna("unknown").astype(str).values,
        cell_class=a.cell_class.fillna("").astype(str).values,
        sub_class=a.cell_sub_class.fillna("").astype(str).values,
        name=name,
        side=a.side.fillna("").astype(str).values,
        nt=a.top_nt.fillna("").astype(str).values,
    )


def label(A, i):
    n = A["name"][i] or A["cell_class"][i] or A["super_class"][i]
    s = A["side"][i]
    return f"{n}{' (' + s[0].upper() + ')' if s else ''}"
