"""Compatibility entrypoint — thin shim over the reproducible pipeline.

Original script (kept so existing docs/commands keep working).
The real logic lives in ml/pipeline/. Run:  python ml/scripts/train_model.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ml.pipeline.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
