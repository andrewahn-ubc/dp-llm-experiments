"""
logging_utils.py

Creates two handlers:
  - Console (stderr): WARNING and above, so only important messages show
    interactively when running on a login node.
  - File (run_id.log in the output dir): DEBUG and above, timestamped,
    with the calling module and line number included.

Usage:
    from fact_check.logging_utils import get_logger
    log = get_logger(__name__, output_dir="/path/to/run/dir", run_id="my_run")
    log.info("Starting epoch %d", epoch)
    log.debug("batch loss: ce=%.4f stab=%.4f", ce, stab)
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, output_dir: str | None = None, run_id: str = "run") -> logging.Logger:
    """
    Return a logger that writes:
      - DEBUG+ to  <output_dir>/<run_id>.log   (timestamped, with file:line)
      - WARNING+ to stderr                      (clean, for interactive use)

    Safe to call multiple times with the same name — handlers are not
    duplicated if the logger already exists.
    """
    logger = logging.getLogger(name)

    # If handlers already attached (e.g. called twice), return as-is.
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't bubble up to root logger

    fmt_file    = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_console = logging.Formatter(
        fmt="%(levelname)-8s | %(message)s",
    )

    # ── File handler ─────────────────────────────────────────────────────────
    if output_dir is not None:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{run_id}.log"

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)

    # ── Console handler (stderr so it doesn't mix with stdout data) ──────────
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)   # INFO+ on console; DEBUG only goes to file
    ch.setFormatter(fmt_console)
    logger.addHandler(ch)

    return logger
