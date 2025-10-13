"""
Budget Analytics and Forecasting

Provides spending analysis and simple linear forecasting.
Zero external dependencies, minimal but effective implementation.

Per SDD.md:
- Minimal, functional implementation
- Simple linear forecasting (no ML needed)
- Spending trends and patterns
- ROI analysis from optimizations
"""

import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .tracker import BudgetTracker

logger = logging.getLogger(__name__)


class BudgetAnalytics:
    """
    Provides analytics and forecasting for budget management.

    Uses simple statistical methods for accurate predictions without
    complex machine learning dependencies.
    """

    def __init__(self, tracker: BudgetTracker):
        """
        Initialize budget analytics.

        Args:
            tracker: Budget tracker instance
        """
        self.tracker = tracker

    def forecast_spending(
        self,
        days_ahead: int = 7,
        historical_days: int = 30,
    ) -> Dict[str, any]:
        """
        Forecast future spending using simple linear projection.

        Args:
            days_ahead: Number of days to forecast
            historical_days: Days of historical data to use

        Returns:
            Dictionary with forecast information
        """
        # Get historical daily spending
        history = self.tracker.get_daily_spending_history(days=historical_days)

        if not history:
            return {
                "error": "Insufficient historical data for forecasting",
                "days_ahead": days_ahead,
                "historical_days": historical_days,
            }

        # Extract daily costs
        daily_costs = [record["total_cost"] for record in history]

        # Calculate statistics
        avg_daily = statistics.mean(daily_costs) if daily_costs else 0.0
        median_daily = statistics.median(daily_costs) if daily_costs else 0.0

        # Calculate variance for confidence interval
        if len(daily_costs) > 1:
            stdev_daily = statistics.stdev(daily_costs)
        else:
            stdev_daily = 0.0

        # Simple linear projection
        projected_total = avg_daily * days_ahead

        # Confidence intervals (±1 standard deviation)
        min_projected = max(0, (avg_daily - stdev_daily) * days_ahead)
        max_projected = (avg_daily + stdev_daily) * days_ahead

        # Detect trend (simple: compare first half to second half)
        if len(daily_costs) >= 4:
            mid = len(daily_costs) // 2
            first_half_avg = statistics.mean(daily_costs[:mid])
            second_half_avg = statistics.mean(daily_costs[mid:])

            if second_half_avg > first_half_avg * 1.1:
                trend = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "unknown"

        return {
            "days_ahead": days_ahead,
            "historical_days": len(daily_costs),
            "avg_daily_cost": avg_daily,
            "median_daily_cost": median_daily,
            "stdev_daily_cost": stdev_daily,
            "projected_total": projected_total,
            "confidence_interval": {
                "min": min_projected,
                "max": max_projected,
            },
            "trend": trend,
            "recommendation": self._generate_recommendation(
                avg_daily, projected_total, trend
            ),
        }

    def _generate_recommendation(
        self,
        avg_daily: float,
        projected_total: float,
        trend: str,
    ) -> str:
        """Generate spending recommendation."""
        if trend == "increasing":
            return f"Spending is trending up. Consider reviewing usage to reduce costs."
        elif trend == "decreasing":
            return f"Spending is trending down. Current optimizations are effective."
        elif avg_daily > 10.0:
            return f"High daily spending (${avg_daily:.2f}/day). Review for optimization opportunities."
        else:
            return f"Spending is stable at ${avg_daily:.2f}/day."

    def analyze_spending_patterns(
        self,
        days: int = 30,
    ) -> Dict[str, any]:
        """
        Analyze spending patterns over time.

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with pattern analysis
        """
        history = self.tracker.get_daily_spending_history(days=days)

        if not history:
            return {"error": "No historical data available"}

        # Find peak spending days
        sorted_by_cost = sorted(
            history,
            key=lambda x: x["total_cost"],
            reverse=True,
        )
        peak_days = sorted_by_cost[:5]

        # Find most expensive models
        spending = self.tracker.get_spending(period="custom", start_time=time.time() - (days * 86400))
        top_models = spending.get("by_model", [])[:5]

        # Calculate cost per request
        total_requests = spending.get("total_requests", 0)
        total_cost = spending.get("total_cost", 0.0)
        cost_per_request = total_cost / total_requests if total_requests > 0 else 0.0

        return {
            "period_days": days,
            "total_cost": total_cost,
            "total_requests": total_requests,
            "cost_per_request": cost_per_request,
            "peak_spending_days": peak_days,
            "top_expensive_models": top_models,
            "by_provider": spending.get("by_provider", []),
        }

    def calculate_savings(
        self,
        baseline_cost: float,
        optimized_cost: float,
    ) -> Dict[str, any]:
        """
        Calculate cost savings from optimizations.

        Args:
            baseline_cost: Original cost without optimization
            optimized_cost: Cost after optimization

        Returns:
            Dictionary with savings information
        """
        savings = baseline_cost - optimized_cost
        savings_percentage = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

        return {
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
            "savings": savings,
            "savings_percentage": savings_percentage,
            "roi": f"{savings_percentage:.1f}% cost reduction",
        }

    def get_cost_breakdown(
        self,
        period: str = "month",
    ) -> Dict[str, any]:
        """
        Get detailed cost breakdown.

        Args:
            period: 'day', 'week', or 'month'

        Returns:
            Dictionary with cost breakdown
        """
        spending = self.tracker.get_spending(period=period)

        # Calculate percentages
        total_cost = spending.get("total_cost", 0.0)
        by_provider = spending.get("by_provider", [])

        for provider in by_provider:
            provider["percentage"] = (
                (provider["total_cost"] / total_cost * 100)
                if total_cost > 0
                else 0.0
            )

        # Token efficiency
        total_input = spending.get("total_input_tokens", 0)
        total_output = spending.get("total_output_tokens", 0)
        total_tokens = total_input + total_output

        cost_per_1k_tokens = (total_cost / total_tokens * 1000) if total_tokens > 0 else 0.0

        return {
            "period": period,
            "total_cost": total_cost,
            "total_requests": spending.get("total_requests", 0),
            "total_tokens": total_tokens,
            "cost_per_request": (
                total_cost / spending.get("total_requests", 1)
                if spending.get("total_requests", 0) > 0
                else 0.0
            ),
            "cost_per_1k_tokens": cost_per_1k_tokens,
            "by_provider": by_provider,
            "by_model": spending.get("by_model", []),
        }

    def compare_periods(
        self,
        period1: str = "week",
        period2: str = "month",
    ) -> Dict[str, any]:
        """
        Compare spending across different time periods.

        Args:
            period1: First period to compare
            period2: Second period to compare

        Returns:
            Dictionary with comparison data
        """
        spending1 = self.tracker.get_spending(period=period1)
        spending2 = self.tracker.get_spending(period=period2)

        cost1 = spending1.get("total_cost", 0.0)
        cost2 = spending2.get("total_cost", 0.0)

        # Calculate daily averages
        days1 = 1 if period1 == "day" else 7 if period1 == "week" else 30
        days2 = 1 if period2 == "day" else 7 if period2 == "week" else 30

        daily_avg1 = cost1 / days1 if days1 > 0 else 0.0
        daily_avg2 = cost2 / days2 if days2 > 0 else 0.0

        return {
            "period1": {
                "name": period1,
                "total_cost": cost1,
                "total_requests": spending1.get("total_requests", 0),
                "daily_average": daily_avg1,
            },
            "period2": {
                "name": period2,
                "total_cost": cost2,
                "total_requests": spending2.get("total_requests", 0),
                "daily_average": daily_avg2,
            },
            "comparison": {
                "cost_difference": cost2 - cost1,
                "daily_avg_difference": daily_avg2 - daily_avg1,
                "percentage_change": (
                    ((cost2 - cost1) / cost1 * 100) if cost1 > 0 else 0.0
                ),
            },
        }

    def generate_report(
        self,
        report_type: str = "summary",
        days: int = 30,
    ) -> Dict[str, any]:
        """
        Generate comprehensive spending report.

        Args:
            report_type: 'summary' or 'detailed'
            days: Number of days to include

        Returns:
            Dictionary with report data
        """
        # Get basic spending info
        monthly = self.tracker.get_spending(period="month")
        daily = self.tracker.get_spending(period="day")

        # Get forecast
        forecast = self.forecast_spending(days_ahead=7, historical_days=days)

        # Get patterns
        patterns = self.analyze_spending_patterns(days=days)

        report = {
            "report_type": report_type,
            "generated_at": datetime.now().isoformat(),
            "period_days": days,
            "current_spending": {
                "today": daily.get("total_cost", 0.0),
                "this_month": monthly.get("total_cost", 0.0),
            },
            "forecast": forecast,
            "top_providers": patterns.get("by_provider", [])[:3],
            "top_models": patterns.get("top_expensive_models", [])[:3],
        }

        if report_type == "detailed":
            report["detailed_patterns"] = patterns
            report["cost_breakdown"] = self.get_cost_breakdown(period="month")

        return report


# Global singleton
_analytics: Optional[BudgetAnalytics] = None


def get_budget_analytics(
    tracker: Optional[BudgetTracker] = None,
) -> BudgetAnalytics:
    """
    Get global budget analytics instance.

    Args:
        tracker: Budget tracker instance

    Returns:
        BudgetAnalytics instance
    """
    global _analytics

    if _analytics is None:
        if tracker is None:
            from .tracker import get_budget_tracker
            tracker = get_budget_tracker()
        _analytics = BudgetAnalytics(tracker)

    return _analytics


def close_budget_analytics() -> None:
    """Close global budget analytics."""
    global _analytics
    _analytics = None
