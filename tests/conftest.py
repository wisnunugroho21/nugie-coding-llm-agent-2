"""Put the repo root on sys.path so `import kimi_linear_gdn2`, `training.*`, etc.
resolve when running the tests from any working directory."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
