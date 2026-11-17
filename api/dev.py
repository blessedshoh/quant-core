"""Local development server.

    python api/dev.py

Starts the API on http://127.0.0.1:8000 and lets the local front end
(http://localhost:4321) call it. Never used in production.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QUANTCORE_DEV_ORIGINS", "1")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
