"""Append-only history for camera Gloss and translation stages."""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class AppendOnlyGlossHistory:
    """Write one JSON record per line without overwriting previous records."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()

    def append(self, event: str, data: Optional[Dict[str, Any]] = None) -> bool:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": str(event),
            "data": data or {},
        }
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
            return True
        except OSError:
            return False


__all__ = ["AppendOnlyGlossHistory"]
