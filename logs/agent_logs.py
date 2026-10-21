"""
Structured Agent Logger — AI Video Director
============================================
Each agent calls these helpers to emit consistent, parseable log events.
Failed runs are serialised to logs/failed_runs/{job_id}_{agent}.json
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


# ── Directory Setup ────────────────────────────────────────────────────────────

FAILED_RUNS_DIR = Path("logs/failed_runs")
FAILED_RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ── Configure Loguru ──────────────────────────────────────────────────────────

def setup_logger(log_level: str = "DEBUG") -> None:
    """Configure loguru to write both to stderr and a rotating file."""
    logger.remove()
    logger.add(
        "logs/ai_director_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        compression="gz",
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]}</cyan> | "
            "<cyan>{extra[job_id]}</cyan> | "
            "{message}"
        ),
        enqueue=True,
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[agent]}</cyan> | "
            "{message}"
        ),
        colorize=True,
    )


setup_logger()


# ── Agent Logger Class ────────────────────────────────────────────────────────

class AgentLogger:
    """
    Contextual logger bound to a specific agent + job.
    Usage:
        log = AgentLogger("scene_agent", job_id)
        log.start("Extracting scenes from 5 clips")
        log.success("Extracted 42 scenes")
        log.failure(exception, retry_count=1)
    """

    def __init__(self, agent_name: str, job_id: str):
        self.agent_name = agent_name
        self.job_id = job_id
        self._bound = logger.bind(agent=agent_name, job_id=job_id)
        self._start_time: Optional[float] = None

    def start(self, message: str, **kwargs: Any) -> None:
        import time
        self._start_time = time.perf_counter()
        self._bound.info(f"▶ START  | {message}", **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._bound.info(f"  ℹ️  {message}", **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._bound.debug(f"  🔍 {message}", **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._bound.warning(f"  ⚠️  {message}", **kwargs)

    def success(self, message: str, **kwargs: Any) -> None:
        import time
        elapsed = (
            (time.perf_counter() - self._start_time) * 1000
            if self._start_time
            else 0
        )
        self._bound.success(
            f"✅ DONE  | {message} [{elapsed:.0f}ms]", **kwargs
        )

    def failure(
        self,
        exc: Exception,
        retry_attempt: int = 0,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a failure and serialise it to failed_runs/."""
        tb = traceback.format_exc()
        self._bound.error(
            f"❌ FAIL  | {type(exc).__name__}: {exc} | retry={retry_attempt}"
        )
        self._bound.debug(f"Traceback:\n{tb}")
        self._dump_failure(exc, retry_attempt, tb, context or {})

    def _dump_failure(
        self,
        exc: Exception,
        retry_attempt: int,
        tb: str,
        context: Dict[str, Any],
    ) -> None:
        """Write a structured JSON failure snapshot for post-mortem analysis."""
        payload = {
            "job_id": self.job_id,
            "agent": self.agent_name,
            "timestamp": datetime.utcnow().isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "retry_attempt": retry_attempt,
            "traceback": tb,
            "context": context,
        }
        fname = (
            FAILED_RUNS_DIR
            / f"{self.job_id}_{self.agent_name}_{retry_attempt}.json"
        )
        try:
            fname.write_text(json.dumps(payload, indent=2, default=str))
            self._bound.debug(f"Failure snapshot → {fname}")
        except Exception as write_err:
            self._bound.error(f"Could not write failure snapshot: {write_err}")

    def elapsed_ms(self) -> float:
        """Return milliseconds since start()."""
        import time
        if self._start_time is None:
            return 0.0
        return (time.perf_counter() - self._start_time) * 1000
