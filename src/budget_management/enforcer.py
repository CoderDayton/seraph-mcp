"""
Budget Enforcer

Checks budget limits before API calls and enforces spending constraints.
Supports soft (warning) and hard (blocking) enforcement modes.

Per SDD.md:
- Minimal, functional implementation
- Soft enforcement: log warnings but allow requests
- Hard enforcement: block requests when budget exceeded
- Real-time budget checking
"""

import logging
import time
from typing import Any

from .config import BudgetConfig, EnforcementMode
from .tracker import BudgetTracker

logger = logging.getLogger(__name__)


class BudgetEnforcer:
    """
    Enforces budget limits with configurable soft/hard modes.

    Soft mode: Logs warnings but allows requests to proceed
    Hard mode: Blocks requests when budget is exceeded
    """

    def __init__(self, config: BudgetConfig, tracker: BudgetTracker):
        """
        Initialize budget enforcer.

        Args:
            config: Budget configuration
            tracker: Budget tracker instance
        """
        self.config = config
        self.tracker = tracker
        self._alert_state: dict[str, float] = {}  # Track alert timestamps for thresholds

    def check_budget(
        self,
        estimated_cost: float | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if budget allows a new request.

        Args:
            estimated_cost: Estimated cost of upcoming request (optional)

        Returns:
            Tuple of (allowed: bool, status: dict)
        """
        if not self.config.enabled:
            return True, {"status": "disabled"}

        # Get current spending
        daily_spend = self.tracker.get_spending(period="day")["total_cost"]
        monthly_spend = self.tracker.get_spending(period="month")["total_cost"]
        weekly_spend = self.tracker.get_spending(period="week")["total_cost"]

        # Include estimated cost in checks
        estimated_cost = estimated_cost or 0.0
        projected_daily = daily_spend + estimated_cost
        projected_monthly = monthly_spend + estimated_cost
        projected_weekly = weekly_spend + estimated_cost

        status = {
            "daily_spend": daily_spend,
            "monthly_spend": monthly_spend,
            "weekly_spend": weekly_spend,
            "estimated_cost": estimated_cost,
            "projected_daily": projected_daily,
            "projected_monthly": projected_monthly,
            "projected_weekly": projected_weekly,
        }

        # Check daily limit
        if self.config.daily_limit is not None:
            if projected_daily > self.config.daily_limit:
                return self._handle_limit_exceeded(
                    "daily",
                    projected_daily,
                    self.config.daily_limit,
                    status,
                )
            # Check thresholds
            self._check_thresholds(
                "daily",
                daily_spend,
                self.config.daily_limit,
            )

        # Check weekly limit
        if self.config.weekly_limit is not None:
            if projected_weekly > self.config.weekly_limit:
                return self._handle_limit_exceeded(
                    "weekly",
                    projected_weekly,
                    self.config.weekly_limit,
                    status,
                )
            self._check_thresholds(
                "weekly",
                weekly_spend,
                self.config.weekly_limit,
            )

        # Check monthly limit
        if self.config.monthly_limit is not None:
            if projected_monthly > self.config.monthly_limit:
                return self._handle_limit_exceeded(
                    "monthly",
                    projected_monthly,
                    self.config.monthly_limit,
                    status,
                )
            self._check_thresholds(
                "monthly",
                monthly_spend,
                self.config.monthly_limit,
            )

        status["status"] = "ok"
        status["enforcement_mode"] = self.config.enforcement_mode
        return True, status

    def _handle_limit_exceeded(
        self,
        period: str,
        current_spend: float,
        limit: float,
        status: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Handle budget limit exceeded."""
        message = f"Budget limit exceeded: {period} spending ${current_spend:.4f} exceeds limit of ${limit:.4f}"

        # Record alert
        self.tracker.record_alert(
            period_type=period,
            threshold=1.0,
            current_spend=current_spend,
            budget_limit=limit,
            message=message,
        )

        if self.config.enforcement_mode == EnforcementMode.HARD:
            # Hard mode: block request
            logger.warning(f"🚫 BLOCKED: {message}")
            status["status"] = "blocked"
            status["reason"] = message
            status["enforcement_mode"] = "hard"
            return False, status
        else:
            # Soft mode: allow but warn
            logger.warning(f"⚠️  WARNING: {message}")
            status["status"] = "warning"
            status["reason"] = message
            status["enforcement_mode"] = "soft"
            return True, status

    def _check_thresholds(
        self,
        period: str,
        current_spend: float,
        limit: float,
    ) -> None:
        """Check alert thresholds and send notifications."""
        percentage = current_spend / limit if limit > 0 else 0

        for threshold in self.config.alert_thresholds:
            # Check if threshold crossed
            if percentage >= threshold:
                # Check if we've already alerted for this threshold
                alert_key = f"{period}:{threshold}"
                if alert_key not in self._alert_state:
                    self._send_threshold_alert(
                        period,
                        threshold,
                        current_spend,
                        limit,
                        percentage,
                    )
                    self._alert_state[alert_key] = time.time()

    def _send_threshold_alert(
        self,
        period: str,
        threshold: float,
        current_spend: float,
        limit: float,
        percentage: float,
    ) -> None:
        """Send alert for threshold crossing."""
        message = f"Budget alert: {period} spending at {percentage * 100:.1f}% (${current_spend:.4f} / ${limit:.4f})"

        # Record alert
        self.tracker.record_alert(
            period_type=period,
            threshold=threshold,
            current_spend=current_spend,
            budget_limit=limit,
            message=message,
        )

        # Log alert
        if percentage >= 0.9:
            logger.warning(f"🔴 {message}")
        elif percentage >= 0.75:
            logger.warning(f"🟡 {message}")
        else:
            logger.info(f"🟢 {message}")

        # Send webhook if configured
        if self.config.webhook_enabled and self.config.webhook_url:
            self._send_webhook_alert(message, period, threshold, current_spend, limit)

    def _send_webhook_alert(
        self,
        message: str,
        period: str,
        threshold: float,
        current_spend: float,
        limit: float,
    ) -> None:
        """Send webhook notification."""
        try:
            import json
            import urllib.request

            if self.config.webhook_url is None:
                raise ValueError("webhook_url must be set for webhook alerts")

            # Validate URL scheme for security (prevent file:/ or custom schemes)
            from urllib.parse import urlparse

            parsed_url = urlparse(self.config.webhook_url)
            if parsed_url.scheme not in ("http", "https"):
                raise ValueError(f"Invalid webhook URL scheme: {parsed_url.scheme}. Only http and https are allowed.")

            payload = {
                "type": "budget_alert",
                "message": message,
                "period": period,
                "threshold": threshold,
                "current_spend": current_spend,
                "budget_limit": limit,
                "percentage": (current_spend / limit) * 100 if limit > 0 else 0,
                "timestamp": time.time(),
            }

            req = urllib.request.Request(
                self.config.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=5) as response:  # nosec B310 - URL scheme validated above
                if response.status == 200:
                    logger.debug("Webhook alert sent successfully")
                else:
                    logger.warning(f"Webhook returned status {response.status}")

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    def reset_alert_state(self) -> None:
        """Reset alert state (e.g., at start of new period)."""
        self._alert_state.clear()
        logger.info("Alert state reset")

    def get_budget_status(self) -> dict[str, Any]:
        """
        Get current budget status.

        Returns:
            Dictionary with budget status information
        """
        daily_spend = self.tracker.get_spending(period="day")["total_cost"]
        monthly_spend = self.tracker.get_spending(period="month")["total_cost"]
        weekly_spend = self.tracker.get_spending(period="week")["total_cost"]

        status = {
            "enabled": self.config.enabled,
            "enforcement_mode": self.config.enforcement_mode,
            "daily": {
                "limit": self.config.daily_limit,
                "spent": daily_spend,
                "remaining": (self.config.daily_limit - daily_spend) if self.config.daily_limit else None,
                "percentage": (daily_spend / self.config.daily_limit * 100) if self.config.daily_limit else None,
            },
            "weekly": {
                "limit": self.config.weekly_limit,
                "spent": weekly_spend,
                "remaining": (self.config.weekly_limit - weekly_spend) if self.config.weekly_limit else None,
                "percentage": (weekly_spend / self.config.weekly_limit * 100) if self.config.weekly_limit else None,
            },
            "monthly": {
                "limit": self.config.monthly_limit,
                "spent": monthly_spend,
                "remaining": (self.config.monthly_limit - monthly_spend) if self.config.monthly_limit else None,
                "percentage": (monthly_spend / self.config.monthly_limit * 100) if self.config.monthly_limit else None,
            },
            "alert_thresholds": self.config.alert_thresholds,
        }

        return status


# Global singleton
_enforcer: BudgetEnforcer | None = None


def get_budget_enforcer(
    config: BudgetConfig | None = None,
    tracker: BudgetTracker | None = None,
) -> BudgetEnforcer:
    """
    Get global budget enforcer instance.

    Args:
        config: Budget configuration
        tracker: Budget tracker instance

    Returns:
        BudgetEnforcer instance
    """
    global _enforcer

    if _enforcer is None:
        if config is None:
            config = BudgetConfig()
        if tracker is None:
            from .tracker import get_budget_tracker

            tracker = get_budget_tracker()
        _enforcer = BudgetEnforcer(config, tracker)

    return _enforcer


def close_budget_enforcer() -> None:
    """Close global budget enforcer."""
    global _enforcer
    _enforcer = None
