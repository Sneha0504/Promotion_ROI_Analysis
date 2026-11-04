"""Typed configuration loader.

Reads ``config/config.yaml`` into nested, type-hinted dataclasses so the rest of the
code accesses tunables through attributes rather than raw dictionaries. All file-system
paths are resolved relative to the project root, which keeps the pipeline working even
though the project lives under a directory whose name contains a space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root = parent of the ``src`` directory holding this file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Directory locations, resolved to absolute paths under the project root."""

    data_raw: Path
    data_processed: Path
    data_interim: Path
    sql: Path
    docs: Path
    docs_site: Path


@dataclass(frozen=True)
class DuckDB:
    """Execution bounds for DuckDB scans of the large CSVs."""

    threads: int
    memory_limit: str
    temp_dir: Path
    max_temp_size: str


@dataclass(frozen=True)
class Files:
    """Raw CSV file names inside ``paths.data_raw``."""

    train: str
    items: str
    stores: str
    oil: str
    holidays: str
    transactions: str


@dataclass(frozen=True)
class Windows:
    """Pre/post window lengths in days."""

    pre_days: int
    post_days: int


@dataclass(frozen=True)
class Selection:
    """Promotion-event selection criteria."""

    min_promo_days: int
    max_promo_days: int
    min_pre_nonzero_frac: float
    promo_flag_gap_tolerance: int
    require_store_open: bool
    max_events: int


@dataclass(frozen=True)
class Donors:
    """Donor-pool criteria for synthetic control."""

    same_family: bool
    same_class: bool
    same_store_first: bool
    expand_to_cluster: bool
    max_donors: int
    min_donors: int
    min_donor_pre_corr: float
    min_donor_pre_nonzero_frac: float


@dataclass(frozen=True)
class ITS:
    """Interrupted-time-series model settings."""

    log_transform: bool
    use_oil: bool
    min_pre_r2: float
    max_pre_nrmse: float
    ljung_box_lags: int
    bootstrap_draws: int
    block_size: int
    cf_cap_multiple: float
    cf_cap_mean_multiple: float


@dataclass(frozen=True)
class Synth:
    """Synthetic-control model settings."""

    smooth_window: int
    allow_intercept: bool
    max_pre_rmspe: float
    solver_max_iter: int
    bootstrap_draws: int
    block_size: int
    placebo_pvalue_threshold: float


@dataclass(frozen=True)
class Economics:
    """Modeled (assumed) price/margin economics; Favorita has no price data."""

    assumed_price: float
    discount_depth: float
    gross_margin: float


@dataclass(frozen=True)
class Significance:
    """Statistical reporting settings."""

    ci_level: float


@dataclass(frozen=True)
class Cannibalization:
    """Cannibalization settings."""

    post_near_zero_units: float


@dataclass(frozen=True)
class Classification:
    """Promotion net-impact classification settings."""

    inconclusive_basis: str


@dataclass(frozen=True)
class Sensitivity:
    """Economic-assumption sensitivity grid for the net-impact sweep."""

    discount_depths: list[float]
    gross_margins: list[float]


@dataclass(frozen=True)
class EarlyWarning:
    """Early-warning rule and backtest split settings."""

    early_days: int
    train_fraction: float
    test_fraction: float
    tune_metric: str
    min_precision_lift: float
    recall_floor: float
    threshold_grid_steps: int
    k_folds: int


@dataclass(frozen=True)
class BreakEven:
    """Break-even economics thresholds."""

    alt_discount_depth: float
    alt_gross_margin: float


@dataclass(frozen=True)
class Config:
    """Top-level typed configuration."""

    paths: Paths
    duckdb: DuckDB
    files: Files
    windows: Windows
    selection: Selection
    donors: Donors
    its: ITS
    synth: Synth
    economics: Economics
    significance: Significance
    cannibalization: Cannibalization
    classification: Classification
    sensitivity: Sensitivity
    break_even: BreakEven
    early_warning: EarlyWarning
    random_seed: int
    reliable_flag_start: str

    def train_path(self) -> Path:
        """Absolute path to the raw ``train.csv``."""
        return self.paths.data_raw / self.files.train

    def raw_file(self, key: str) -> Path:
        """Absolute path to a raw file by its config key (e.g. ``'items'``)."""
        return self.paths.data_raw / getattr(self.files, key)


def _resolve(root: Path, value: str) -> Path:
    """Resolve ``value`` to an absolute path relative to ``root`` if not absolute."""
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate the YAML configuration into a :class:`Config`.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully populated, frozen :class:`Config` instance.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    paths = Paths(
        data_raw=_resolve(PROJECT_ROOT, raw["paths"]["data_raw"]),
        data_processed=_resolve(PROJECT_ROOT, raw["paths"]["data_processed"]),
        data_interim=_resolve(PROJECT_ROOT, raw["paths"]["data_interim"]),
        sql=_resolve(PROJECT_ROOT, raw["paths"]["sql"]),
        docs=_resolve(PROJECT_ROOT, raw["paths"]["docs"]),
        docs_site=_resolve(PROJECT_ROOT, raw["paths"]["docs_site"]),
    )

    duckdb_cfg = DuckDB(
        threads=int(raw["duckdb"]["threads"]),
        memory_limit=str(raw["duckdb"]["memory_limit"]),
        temp_dir=_resolve(PROJECT_ROOT, raw["duckdb"]["temp_dir"]),
        max_temp_size=str(raw["duckdb"]["max_temp_size"]),
    )

    return Config(
        paths=paths,
        duckdb=duckdb_cfg,
        files=Files(**raw["files"]),
        windows=Windows(**raw["windows"]),
        selection=Selection(**raw["selection"]),
        donors=Donors(**raw["donors"]),
        its=ITS(**raw["its"]),
        synth=Synth(**raw["synth"]),
        economics=Economics(**raw["economics"]),
        significance=Significance(**raw["significance"]),
        cannibalization=Cannibalization(**raw["cannibalization"]),
        classification=Classification(**raw["classification"]),
        sensitivity=Sensitivity(
            discount_depths=[float(x) for x in raw["sensitivity"]["discount_depths"]],
            gross_margins=[float(x) for x in raw["sensitivity"]["gross_margins"]],
        ),
        break_even=BreakEven(**raw["break_even"]),
        early_warning=EarlyWarning(**raw["early_warning"]),
        random_seed=int(raw["random_seed"]),
        reliable_flag_start=str(raw["reliable_flag_start"]),
    )
