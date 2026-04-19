"""
config_loader.py — Load fact_check/config.yaml via OmegaConf.

Path values use OmegaConf's oc.env resolver:
  ${oc.env:FC_VAR}            — required; raises if FC_VAR is not set
  ${oc.env:FC_VAR,/fallback}  — optional; uses /fallback if FC_VAR is not set

Source env.sh before running any script:
    source env.sh fact-check

Usage:
    from fact_check.config_loader import load_config

    cfg = load_config()
    print(cfg.paths.data_dir)   # resolved from $FC_DATA_DIR
    print(cfg.model.name)
"""

from pathlib import Path
from omegaconf import OmegaConf, DictConfig

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: Path | None = None) -> DictConfig:
    return OmegaConf.load(path or _CONFIG_PATH)
