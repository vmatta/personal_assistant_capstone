"""Structured logging for the harness; each event is also written as JSONL
so later evaluation runs can compute latency, escalation rate, etc."""

import json
import logging
import time
from pathlib import Path

from config import SETTINGS

_HARNESS_CFG = SETTINGS.get("harness", {})
_LOG_DIR = Path("data/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "run_events.jsonl"

logger = logging.getLogger("personal_assistant")
logger.setLevel(logging.DEBUG if _HARNESS_CFG.get("debug_mode", False) else logging.INFO)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_handler)


def log_event(event_type: str, **fields) -> None:
    """Append a structured JSON event to the run log and echo it to the console."""
    record = {"timestamp": time.time(), "event": event_type, **fields}
    with _LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.debug("%s | %s", event_type, fields)
