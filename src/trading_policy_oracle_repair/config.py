from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    data_root: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")
    checkpoints_dir: Path = Path("checkpoints")


class RawDataConfig(BaseModel):
    provider: Literal["yfinance"] = "yfinance"
    interval: str = "1d"
    start: str = "2005-01-01"
    end: str | None = None
    auto_adjust: bool = False
    actions: bool = True
    repair: bool = False
    keepna: bool = False
    timeout_seconds: int = Field(default=30, ge=1, le=120)
    tickers: list[str]
    cash_proxy: str = "^IRX"

    @field_validator("tickers")
    @classmethod
    def tickers_nonempty_unique(cls, value: list[str]) -> list[str]:
        cleaned = [x.strip().upper() for x in value if x.strip()]
        if not cleaned:
            raise ValueError("At least one ticker is required.")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Duplicate tickers are not allowed.")
        return cleaned

    @model_validator(mode="after")
    def enforce_research_download_contract(self):
        if self.auto_adjust:
            raise ValueError("Raw data requires auto_adjust=false to preserve raw OHLC.")
        if not self.actions:
            raise ValueError("Raw data requires actions=true to preserve dividends/splits.")
        return self


class ResearchContract(BaseModel):
    signal_time: Literal["close"] = "close"
    execution_time: Literal["next_open"] = "next_open"
    cost_bps_per_side: float = Field(default=5.0, ge=0)
    horizon_days: int = Field(default=60, ge=1)
    embargo_days: int = Field(default=5, ge=0)
    start_exposures: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    initial_train_years: int = Field(default=10, ge=1)
    calibration_years: int = Field(default=1, ge=1)
    test_years: int = Field(default=1, ge=1)
    bounded_switch_k_per_year: int = Field(default=10, ge=0)
    publication_data_quality_required: bool = False

    @field_validator("start_exposures")
    @classmethod
    def exposures_valid(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("start_exposures may not be empty")
        if any((x < 0.0 or x > 1.0) for x in value):
            raise ValueError("start_exposures must lie in [0, 1]")
        return value


class Phase0Config(BaseModel):
    create_demo_raw_data: bool = True
    write_csv_copy: bool = True
    run_pytest: bool = True


class Phase1Config(BaseModel):
    require_phase0_checkpoint: bool = True
    raw_provider_subdir: str = "yfinance"
    processed_subdir: str = "phase1"
    write_csv_copy: bool = True
    run_pytest: bool = True
    cash_annualization_days: int = Field(default=252, ge=1)
    total_return_base: float = Field(default=1.0, gt=0)
    cash_method: Literal["effective_annual_252"] = "effective_annual_252"
    price_basis: Literal["vendor_split_adjusted", "raw_native"] = "vendor_split_adjusted"


class Phase2Config(BaseModel):
    require_phase1_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    processed_subdir: str = "phase2"
    initial_wealth: float = Field(default=10000.0, gt=0)
    initial_exposure: float = Field(default=0.0, ge=0.0, le=1.0)
    smoke_asset: str = "AAPL"
    smoke_rows: int = Field(default=80, ge=5)
    write_csv_copy: bool = True
    run_pytest: bool = True




class Phase3Config(BaseModel):
    require_phase2_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    processed_subdir: str = "phase3"
    policies_config: Path = Path("config/policies.yaml")
    smoke_asset: str = "AAPL"
    smoke_rows: int = Field(default=320, ge=130)
    initial_exposure: float = Field(default=0.0, ge=0.0, le=1.0)
    write_csv_copy: bool = True
    run_pytest: bool = True

class Phase4Config(BaseModel):
    require_phase3_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    processed_subdir: str = "phase4"
    policies_config: Path = Path("config/policies.yaml")
    initial_wealth: float = Field(default=10000.0, gt=0)
    initial_exposure: float = Field(default=0.0, ge=0.0, le=1.0)
    primary_cost_bps: float = Field(default=5.0, ge=0.0)
    sensitivity_cost_bps: list[float] = [0.0, 5.0, 20.0]
    sensitivity_asset: str = "AAPL"
    annualization_days: int = Field(default=252, ge=1)
    write_csv_summary: bool = True
    run_pytest: bool = True

    @field_validator("sensitivity_cost_bps")
    @classmethod
    def sensitivity_costs_valid(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("sensitivity_cost_bps may not be empty")
        if any(x < 0 for x in value):
            raise ValueError("sensitivity costs must be nonnegative")
        return [float(x) for x in value]


class Phase5Config(BaseModel):
    require_phase4_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    phase4_processed_subdir: str = "phase4"
    processed_subdir: str = "phase5"
    initial_wealth: float = Field(default=10000.0, gt=0)
    initial_position: int = Field(default=0, ge=0, le=1)
    primary_cost_bps: float = Field(default=5.0, ge=0.0)
    annualization_days: int = Field(default=252, ge=1)
    local_horizon_days: int = Field(default=60, ge=1)
    local_smoke_asset: str = "AAPL"
    local_smoke_points: int = Field(default=5, ge=1, le=20)
    write_csv_summary: bool = True
    run_pytest: bool = True


class Phase6Config(BaseModel):
    require_phase5_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    phase4_processed_subdir: str = "phase4"
    processed_subdir: str = "phase6"
    policies_config: Path = Path("config/policies.yaml")
    horizon_days: int = Field(default=60, ge=1)
    primary_cost_bps: float = Field(default=5.0, ge=0.0)
    annualization_days: int = Field(default=252, ge=1)
    write_csv_sample: bool = True
    csv_sample_rows: int = Field(default=500, ge=0, le=10000)
    run_pytest: bool = True


class Phase7Config(BaseModel):
    require_phase6_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    phase6_processed_subdir: str = "phase6"
    processed_subdir: str = "phase7"
    initial_train_years: int = Field(default=10, ge=1)
    calibration_years: int = Field(default=1, ge=1)
    test_years: int = Field(default=1, ge=1)
    embargo_days: int = Field(default=5, ge=0)
    asset_oos_fraction: float = Field(default=0.30, gt=0.0, lt=1.0)
    require_complete_test_year: bool = True
    annualization_days: int = Field(default=252, ge=1)
    write_csv_summaries: bool = True
    run_pytest: bool = True


class Phase8Config(BaseModel):
    require_phase7_checkpoint: bool = True
    phase7_processed_subdir: str = "phase7"
    processed_subdir: str = "phase8"
    high_regret_quantile: float = Field(default=0.90, gt=0.5, lt=1.0)
    ridge_alpha: float = Field(default=1.0, gt=0.0)
    gbt_learning_rate: float = Field(default=0.05, gt=0.0, le=1.0)
    gbt_max_iter: int = Field(default=80, ge=10, le=500)
    gbt_max_leaf_nodes: int = Field(default=15, ge=2, le=255)
    gbt_min_samples_leaf: int = Field(default=50, ge=2)
    gbt_l2_regularization: float = Field(default=0.001, ge=0.0)
    logistic_c: float = Field(default=1.0, gt=0.0)
    logistic_max_iter: int = Field(default=300, ge=50)
    prediction_splits: list[str] = ["calibration", "same_asset_test", "asset_oos_test"]
    write_csv_summaries: bool = True
    prediction_sample_rows: int = Field(default=1000, ge=0, le=10000)
    run_pytest: bool = True

    @field_validator("prediction_splits")
    @classmethod
    def prediction_splits_valid(cls, value: list[str]) -> list[str]:
        allowed = {"calibration", "same_asset_test", "asset_oos_test"}
        cleaned = list(dict.fromkeys(value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError(f"prediction_splits must be a nonempty subset of {sorted(allowed)}")
        return cleaned


class Phase9Config(BaseModel):
    require_phase8_checkpoint: bool = True
    phase8_processed_subdir: str = "phase8"
    processed_subdir: str = "phase9"
    alpha: float = Field(default=0.10, gt=0.0, lt=0.5)
    utility_models: list[str] = ["ridge", "gradient_boosting"]
    primary_lcb_method: Literal["empirical", "block_bootstrap"] = "empirical"
    deploy_margin: float = 0.0
    frontier_margins: list[float] = [0.0, 0.0025, 0.005, 0.01, 0.02]
    bootstrap_block_length_dates: int = Field(default=20, ge=2)
    bootstrap_resamples: int = Field(default=100, ge=0, le=2000)
    bootstrap_conservative_quantile: float = Field(default=0.10, gt=0.0, lt=0.5)
    min_calibration_rows: int = Field(default=100, ge=20)
    evaluation_splits: list[str] = ["same_asset_test", "asset_oos_test"]
    write_csv_summaries: bool = True
    evidence_sample_rows: int = Field(default=1000, ge=0, le=10000)
    run_pytest: bool = True

    @field_validator("utility_models")
    @classmethod
    def utility_models_valid(cls, value: list[str]) -> list[str]:
        allowed = {"ridge", "gradient_boosting"}
        cleaned = list(dict.fromkeys(value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError(f"utility_models must be a nonempty subset of {sorted(allowed)}")
        return cleaned

    @field_validator("frontier_margins")
    @classmethod
    def frontier_margins_valid(cls, value: list[float]) -> list[float]:
        cleaned = sorted(set(float(x) for x in value))
        if not cleaned or cleaned[0] < 0.0:
            raise ValueError("frontier_margins must be nonempty and nonnegative")
        return cleaned

    @field_validator("evaluation_splits")
    @classmethod
    def evaluation_splits_valid(cls, value: list[str]) -> list[str]:
        allowed = {"same_asset_test", "asset_oos_test"}
        cleaned = list(dict.fromkeys(value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError(f"evaluation_splits must be a nonempty subset of {sorted(allowed)}")
        return cleaned


class Phase10Config(BaseModel):
    require_phase9_checkpoint: bool = True
    phase1_processed_subdir: str = "phase1"
    phase7_processed_subdir: str = "phase7"
    phase8_processed_subdir: str = "phase8"
    phase9_processed_subdir: str = "phase9"
    processed_subdir: str = "phase10"
    policies_config: Path = Path("config/policies.yaml")
    utility_models: list[str] = ["ridge", "gradient_boosting"]
    lcb_methods: list[str] = ["empirical"]
    primary_lcb_method: Literal["empirical", "block_bootstrap"] = "empirical"
    deploy_margin: float = 0.0
    meta_initial_policy_id: str = "buy_hold"
    meta_switch_penalty_bps: float = Field(default=5.0, ge=0.0)
    initial_wealth: float = Field(default=1.0, gt=0.0)
    initial_exposure: float = Field(default=0.0, ge=0.0, le=1.0)
    annualization_days: int = Field(default=252, ge=1)
    evaluation_horizon_days: int = Field(default=60, ge=1)
    evaluation_splits: list[str] = ["same_asset_test", "asset_oos_test"]
    include_naive_meta: bool = True
    run_continuous_pseudo_live: bool = True
    write_csv_summaries: bool = True
    daily_sample_rows: int = Field(default=1000, ge=0, le=10000)
    run_pytest: bool = True

    @field_validator("utility_models")
    @classmethod
    def phase10_models_valid(cls, value: list[str]) -> list[str]:
        allowed = {"ridge", "gradient_boosting"}
        cleaned = list(dict.fromkeys(value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError(f"utility_models must be a nonempty subset of {sorted(allowed)}")
        return cleaned

    @field_validator("lcb_methods")
    @classmethod
    def lcb_methods_valid(cls, value: list[str]) -> list[str]:
        allowed = {"empirical", "block_bootstrap"}
        cleaned = list(dict.fromkeys(value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError(f"lcb_methods must be a nonempty subset of {sorted(allowed)}")
        return cleaned

    @field_validator("evaluation_splits")
    @classmethod
    def phase10_splits_valid(cls, value: list[str]) -> list[str]:
        allowed = {"same_asset_test", "asset_oos_test"}
        cleaned = list(dict.fromkeys(value))
        if not cleaned or any(x not in allowed for x in cleaned):
            raise ValueError(f"evaluation_splits must be a nonempty subset of {sorted(allowed)}")
        return cleaned


class Phase11Config(BaseModel):
    require_phase10_checkpoint: bool = True
    phase10_processed_subdir: str = "phase10"
    processed_subdir: str = "phase11"
    bootstrap_block_length_days: int = Field(default=20, ge=2)
    bootstrap_resamples: int = Field(default=1000, ge=100, le=10000)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    annualization_days: int = Field(default=252, ge=1)
    include_continuous_inference: bool = True
    write_csv_summaries: bool = True
    run_pytest: bool = True


class Phase12Config(BaseModel):
    require_phase11_checkpoint: bool = True
    phase8_processed_subdir: str = "phase8"
    phase9_processed_subdir: str = "phase9"
    phase10_processed_subdir: str = "phase10"
    phase11_processed_subdir: str = "phase11"
    processed_subdir: str = "phase12"
    publication_output_subdir: str = "publication_bundle_phase12"
    publication_zip_stem: str = "publication_bundle_phase12"
    synthetic_worlds: list[str] = ["recoverable", "clairvoyance_only"]
    synthetic_signal_strengths: list[float] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    synthetic_replicates: int = Field(default=12, ge=2, le=200)
    synthetic_n_days: int = Field(default=3600, ge=1000, le=50000)
    synthetic_horizon_days: int = Field(default=20, ge=1, le=252)
    synthetic_cost_bps_per_side: float = Field(default=2.0, ge=0.0, le=100.0)
    synthetic_train_fraction: float = Field(default=0.60, gt=0.4, lt=0.8)
    synthetic_calibration_fraction: float = Field(default=0.20, ge=0.1, lt=0.4)
    synthetic_alpha: float = Field(default=0.10, gt=0.0, lt=0.5)
    synthetic_ridge_alpha: float = Field(default=1.0, gt=0.0)
    max_nonrecoverable_coverage: float = Field(default=0.05, ge=0.0, le=0.5)
    write_pdf_figures: bool = True
    write_latex_tables: bool = True
    continuous_figure_baselines: list[str] = ["donchian_55_20", "mom_126", "sma_20_100", "voltarget_15_20"]
    continuous_figure_models: list[str] = ["ridge", "gradient_boosting"]
    continuous_figure_lcb_method: str = "empirical"
    run_pytest: bool = True

    @field_validator("synthetic_worlds")
    @classmethod
    def synthetic_worlds_valid(cls, value: list[str]) -> list[str]:
        allowed = {"recoverable", "clairvoyance_only"}
        cleaned = list(dict.fromkeys(value))
        if set(cleaned) != allowed:
            raise ValueError("synthetic_worlds must contain recoverable and clairvoyance_only")
        return cleaned

    @field_validator("synthetic_signal_strengths")
    @classmethod
    def synthetic_signal_strengths_valid(cls, value: list[float]) -> list[float]:
        cleaned = sorted(set(float(x) for x in value))
        if len(cleaned) < 3 or cleaned[0] < 0.0 or cleaned[-1] <= 0.0:
            raise ValueError("synthetic_signal_strengths must contain >=3 nonnegative values including a positive high-signal point")
        return cleaned

    @model_validator(mode="after")
    def phase12_fraction_contract(self):
        if self.synthetic_train_fraction + self.synthetic_calibration_fraction >= 0.9:
            raise ValueError("synthetic train+calibration fractions must leave at least 10% for test")
        return self


class AppConfig(BaseModel):
    project_name: str
    phase: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] = 0
    seed: int
    paths: PathsConfig
    raw_data: RawDataConfig
    research_contract: ResearchContract
    phase0: Phase0Config | None = None
    phase1: Phase1Config | None = None
    phase2: Phase2Config | None = None
    phase3: Phase3Config | None = None
    phase4: Phase4Config | None = None
    phase5: Phase5Config | None = None
    phase6: Phase6Config | None = None
    phase7: Phase7Config | None = None
    phase8: Phase8Config | None = None
    phase9: Phase9Config | None = None
    phase10: Phase10Config | None = None
    phase11: Phase11Config | None = None
    phase12: Phase12Config | None = None

    @model_validator(mode="after")
    def validate_phase_section(self):
        if self.phase == 0 and self.phase0 is None:
            raise ValueError("phase0 section is required when phase=0")
        if self.phase == 1 and self.phase1 is None:
            raise ValueError("phase1 section is required when phase=1")
        if self.phase == 2 and self.phase2 is None:
            raise ValueError("phase2 section is required when phase=2")
        if self.phase == 3 and self.phase3 is None:
            raise ValueError("phase3 section is required when phase=3")
        if self.phase == 4 and self.phase4 is None:
            raise ValueError("phase4 section is required when phase=4")
        if self.phase == 5 and self.phase5 is None:
            raise ValueError("phase5 section is required when phase=5")
        if self.phase == 6 and self.phase6 is None:
            raise ValueError("phase6 section is required when phase=6")
        if self.phase == 7 and self.phase7 is None:
            raise ValueError("phase7 section is required when phase=7")
        if self.phase == 8 and self.phase8 is None:
            raise ValueError("phase8 section is required when phase=8")
        if self.phase == 9 and self.phase9 is None:
            raise ValueError("phase9 section is required when phase=9")
        if self.phase == 10 and self.phase10 is None:
            raise ValueError("phase10 section is required when phase=10")
        if self.phase == 11 and self.phase11 is None:
            raise ValueError("phase11 section is required when phase=11")
        if self.phase == 12 and self.phase12 is None:
            raise ValueError("phase12 section is required when phase=12")
        return self


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)


def dump_config_dict(config: AppConfig) -> dict:
    return config.model_dump(mode="json")
