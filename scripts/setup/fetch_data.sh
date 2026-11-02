#!/bin/bash
# Everything the model is built from, into external/ (~1.3 GB) and the Hugging Face cache (~9 GB).
#
#     bash scripts/setup/fetch_data.sh            # the connectome, annotations, Shiu's model and Qwen3.5-4B
#     bash scripts/setup/fetch_data.sh --flybody  # also flybody, to re-pack the viewer's fly (viz/cns is already packed)
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p external/male-cns

# Janelia's male CNS v1.0 (Berg et al. 2026), https://male-cns.janelia.org/download/
MCNS=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather \
         connectome-weights-male-cns-v1.0-minconf-0.5.feather; do
  [ -f "external/male-cns/$f" ] || curl -fL --retry 3 -o "external/male-cns/$f" "$MCNS/$f"
done

# FlyWire v783 annotations (Schlegel et al. 2024), for Shiu et al.'s figures on FlyWire
[ -f external/flywire_annotations_v783.tsv ] || curl -fL --retry 3 -o external/flywire_annotations_v783.tsv \
  https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv

# Shiu et al. 2024's model: its constants, its FlyWire connectivity and its Brian2 code (the reference checks)
[ -d external/Drosophila_brain_model ] || git clone --depth 1 https://github.com/philshiu/Drosophila_brain_model.git external/Drosophila_brain_model

# flybody (Vaxenburg et al. 2025, Apache-2.0): the fly's mesh, packed into viz/cns by scripts/setup/pack_flybody.py
if [ "${1:-}" = "--flybody" ]; then
  [ -d external/flybody ] || git clone --depth 1 https://github.com/TuragaLab/flybody.git external/flybody
fi

# Qwen3.5-4B, into the Hugging Face cache (the code loads it with local_files_only)
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("Qwen/Qwen3.5-4B"))'
