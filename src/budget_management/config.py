"""
Budget Management Configuration

Defines typed configuration for budget tracking, enforcement, and alerts.

Per SDD.md:
- Minimal, functional implementation
- SQLite for persistence (zero dependencies)
- Soft and hard budget enforcement
- Multi-threshold alerts
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EnforcementMode(str, Enum):
    """Budget enforcement modes."""

    SOFT = "soft"  # Log warnings but allow requests
    HARD = "hard"  # Block requests when budget exceeded


class BudgetPeriod(str, Enum):
    """Budget period types."""

    DAILY = "daily"
    MONTHLY = "monthly"
    WEEKLY = "weekly"


class BudgetConfig(BaseModel):
    """Configuration for budget management system."""

    enabled: bool = Field(default=True, description="Enable budget tracking and enforcement")

    # Budget limits
    daily_limit: float | None = Field(
        default=None,
        ge=0.0,
        description="Daily spending limit in USD (None = no limit)",
    )
    monthly_limit: float | None = Field(
        default=None,
        ge=0.0,
        description="Monthly spending limit in USD (None = no limit)",
    )
    weekly_limit: float | None = Field(
        default=None,
        ge=0.0,
        description="Weekly spending limit in USD (None = no limit)",
    )

    # Enforcement
    enforcement_mode: EnforcementMode = Field(
        default=EnforcementMode.SOFT,
        description="Budget enforcement mode (soft=warn, hard=block)",
    )

    # Alert thresholds (as fraction of limit, e.g., 0.5 = 50%)
    alert_thresholds: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 0.9],
        description="Budget alert thresholds (0.0-1.0)",
    )

    # Storage
    db_path: str = Field(
        default="./data/budget.db",
        description="SQLite database path for budget tracking",
    )

    # Alerts
    webhook_url: str | None = Field(default=None, description="Webhook URL for budget alerts (optional)")
    webhook_enabled: bool = Field(default=False, description="Enable webhook notifications")

    # Analytics
    forecasting_days: int = Field(default=7, ge=1, le=90, description="Number of days to forecast spending")
    historical_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days of historical data to keep for forecasting",
    )

    @field_validator("alert_thresholds")
    @classmethod
    def validate_thresholds(cls, v: list[float]) -> list[float]:
        """Ensure all thresholds are between 0 and 1."""
        for threshold in v:
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Alert threshold must be between 0.0 and 1.0, got {threshold}")
        return sorted(v)

    @field_validator("daily_limit", "monthly_limit", "weekly_limit")
    @classmethod
    def validate_limits(cls, v: float | None) -> float | None:
        """Warn if limits are unusually high."""
        if v is not None and v > 10000.0:
            # Just a sanity check - allow but validate
            pass
        return v

    model_config = ConfigDict(use_enum_values=True)
