"""
Budget Management Module

Provides comprehensive budget tracking, enforcement, and analytics for API costs.
Zero external dependencies (uses SQLite), minimal but highly functional.

Per SDD.md:
- Minimal, functional implementation
- SQLite for persistence (built into Python)
- Real-time cost tracking per API call
- Soft/hard budget enforcement
- Simple linear forecasting
- Spending analytics and reports

Public API:
    - BudgetTracker: SQLite-based cost tracking
    - BudgetEnforcer: Budget checking and enforcement
    - BudgetAnalytics: Spending analysis and forecasting
    - BudgetConfig: Configuration schema
    - get_budget_tracker(): Get global tracker instance
    - get_budget_enforcer(): Get global enforcer instance
    - get_budget_analytics(): Get global analytics instance

Usage:
    >>> from src.budget_management import (
    ...     get_budget_tracker,
    ...     get_budget_enforcer,
    ...     BudgetConfig,
    ... )
    >>>
    >>> # Configure budget
    >>> config = BudgetConfig(
    ...     enabled=True,
    ...     daily_limit=10.0,
    ...     monthly_limit=200.0,
    ...     enforcement_mode="soft",
    ... )
    >>>
    >>> # Track API cost
    >>> tracker = get_budget_tracker()
    >>> tracker.track_cost(
    ...     provider="openai",
    ...     model="gpt-4",
    ...     input_tokens=1000,
    ...     output_tokens=500,
    ...     cost_usd=0.045,
    ... )
    >>>
    >>> # Check budget before request
    >>> enforcer = get_budget_enforcer(config=config)
    >>> allowed, status = enforcer.check_budget(estimated_cost=0.05)
    >>> if not allowed:
    ...     print(f"Request blocked: {status['reason']}")
    >>>
    >>> # Get spending report
    >>> analytics = get_budget_analytics()
    >>> report = analytics.generate_report(report_type="summary")
    >>> print(f"Monthly spend: ${report['current_spending']['this_month']:.2f}")
    >>>
    >>> # Forecast spending
    >>> forecast = analytics.forecast_spending(days_ahead=7)
    >>> print(f"Projected 7-day cost: ${forecast['projected_total']:.2f}")

Integration with Provider System:
    The budget system integrates seamlessly with the provider system:

    >>> from src.providers import create_provider, ProviderConfig
    >>> from src.budget_management import get_budget_tracker, get_budget_enforcer
    >>>
    >>> # Setup
    >>> tracker = get_budget_tracker()
    >>> enforcer = get_budget_enforcer()
    >>>
    >>> # Before making API call
    >>> allowed, status = enforcer.check_budget(estimated_cost=0.01)
    >>> if not allowed:
    ...     raise RuntimeError(f"Budget exceeded: {status['reason']}")
    >>>
    >>> # Make API call
    >>> provider = create_provider("openai", ProviderConfig(api_key="..."))
    >>> response = await provider.complete(request)
    >>>
    >>> # Track actual cost
    >>> tracker.track_cost(
    ...     provider=response.provider,
    ...     model=response.model,
    ...     input_tokens=response.usage["prompt_tokens"],
    ...     output_tokens=response.usage["completion_tokens"],
    ...     cost_usd=response.cost_usd,
    ... )
"""

from .analytics import BudgetAnalytics, get_budget_analytics, close_budget_analytics
from .config import BudgetConfig, BudgetPeriod, EnforcementMode
from .enforcer import BudgetEnforcer, get_budget_enforcer, close_budget_enforcer
from .tracker import BudgetTracker, get_budget_tracker, close_budget_tracker

__all__ = [
    # Configuration
    "BudgetConfig",
    "BudgetPeriod",
    "EnforcementMode",
    # Core components
    "BudgetTracker",
    "BudgetEnforcer",
    "BudgetAnalytics",
    # Factory functions
    "get_budget_tracker",
    "get_budget_enforcer",
    "get_budget_analytics",
    # Cleanup
    "close_budget_tracker",
    "close_budget_enforcer",
    "close_budget_analytics",
]
