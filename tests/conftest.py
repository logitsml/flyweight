"""CPU tests: python -m pytest tests (CUDA_VISIBLE_DEVICES= to keep them off the GPU)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
