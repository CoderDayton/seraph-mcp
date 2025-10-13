"""
Budget Tracker

SQLite-based cost tracking system for API usage monitoring.
Zero external dependencies, minimal but complete implementation.

Per SDD.md:
- SQLite for persistence (built into Python)
- Real-time cost tracking per API call
- Granular tracking (provider, model, operation)
- Query support for analytics
"""

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BudgetTracker:
    """
    SQLite-based budget tracker for API costs.

    Tracks every API call with provider, model, tokens, and cost information.
    Provides queries for current spending, historical analysis, and forecasting.
    """

    def __init__(self, db_path: str = "./data/budget.db"):
        """
        Initialize budget tracker.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_database()

    def _ensure_db_directory(self) -> None:
        """Ensure database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            # Cost records table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation TEXT DEFAULT 'completion',
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Budget configurations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_type TEXT NOT NULL,
                    limit_usd REAL NOT NULL,
                    alert_thresholds TEXT NOT NULL,
                    enforcement_mode TEXT DEFAULT 'soft',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Budget alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS budget_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    period_type TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    current_spend REAL NOT NULL,
                    budget_limit REAL NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_records_timestamp
                ON cost_records(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_records_provider
                ON cost_records(provider)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cost_records_model
                ON cost_records(model)
            """)

            conn.commit()
            logger.info(f"Budget tracker database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
        finally:
            conn.close()

    def track_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        operation: str = "completion",
        metadata: Optional[str] = None,
    ) -> bool:
        """
        Track a cost record.

        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD
            operation: Operation type (default: 'completion')
            metadata: Optional JSON metadata

        Returns:
            True if successful
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO cost_records
                (timestamp, provider, model, operation, input_tokens, output_tokens, cost_usd, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                provider,
                model,
                operation,
                input_tokens,
                output_tokens,
                cost_usd,
                metadata,
            ))
            conn.commit()
            logger.debug(f"Tracked cost: {provider}/{model} = ${cost_usd:.6f}")
            return True
        except Exception as e:
            logger.error(f"Failed to track cost: {e}")
            return False
        finally:
            conn.close()

    def get_spending(
        self,
        period: str = "day",
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Get spending for a time period.

        Args:
            period: 'day', 'week', 'month', or 'custom'
            start_time: Custom start timestamp (for period='custom')
            end_time: Custom end timestamp (for period='custom')

        Returns:
            Dictionary with spending information
        """
        # Calculate time range
        now = time.time()
        if period == "day":
            start = now - 86400  # 24 hours
        elif period == "week":
            start = now - 604800  # 7 days
        elif period == "month":
            start = now - 2592000  # 30 days
        elif period == "custom":
            start = start_time or now - 86400
            now = end_time or now
        else:
            start = now - 86400  # Default to day

        conn = self._get_connection()
        try:
            # Get total spending
            result = conn.execute("""
                SELECT
                    COUNT(*) as request_count,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    MIN(timestamp) as first_request,
                    MAX(timestamp) as last_request
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start, now)).fetchone()

            # Get spending by provider
            by_provider = conn.execute("""
                SELECT
                    provider,
                    COUNT(*) as request_count,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY provider
                ORDER BY total_cost DESC
            """, (start, now)).fetchall()

            # Get spending by model
            by_model = conn.execute("""
                SELECT
                    provider,
                    model,
                    COUNT(*) as request_count,
                    SUM(cost_usd) as total_cost
                FROM cost_records
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY provider, model
                ORDER BY total_cost DESC
                LIMIT 10
            """, (start, now)).fetchall()

            return {
                "period": period,
                "start_time": start,
                "end_time": now,
                "total_requests": result["request_count"] or 0,
                "total_cost": result["total_cost"] or 0.0,
                "total_input_tokens": result["total_input_tokens"] or 0,
                "total_output_tokens": result["total_output_tokens"] or 0,
                "first_request": result["first_request"],
                "last_request": result["last_request"],
                "by_provider": [dict(row) for row in by_provider],
                "by_model": [dict(row) for row in by_model],
            }
        except Exception as e:
            logger.error(f"Failed to get spending: {e}")
            return {"error": str(e)}
        finally:
            conn.close()

    def get_daily_spending_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get daily spending for the last N days.

        Args:
            days: Number of days to retrieve

        Returns:
            List of daily spending records
        """
        conn = self._get_connection()
        try:
            start_time = time.time() - (days * 86400)

            results = conn.execute("""
                SELECT
                    DATE(datetime(timestamp, 'unixepoch')) as date,
                    COUNT(*) as request_count,
                    SUM(cost_usd) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM cost_records
                WHERE timestamp >= ?
                GROUP BY date
                ORDER BY date ASC
            """, (start_time,)).fetchall()

            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get daily history: {e}")
            return []
        finally:
            conn.close()

    def record_alert(
        self,
        period_type: str,
        threshold: float,
        current_spend: float,
        budget_limit: float,
        message: str,
    ) -> bool:
        """
        Record a budget alert.

        Args:
            period_type: Budget period ('daily', 'monthly', etc.)
            threshold: Threshold that triggered alert (0.0-1.0)
            current_spend: Current spending amount
            budget_limit: Budget limit amount
            message: Alert message

        Returns:
            True if successful
        """
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO budget_alerts
                (timestamp, period_type, threshold, current_spend, budget_limit, message)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                time.time(),
                period_type,
                threshold,
                current_spend,
                budget_limit,
                message,
            ))
            conn.commit()
            logger.info(f"Budget alert recorded: {message}")
            return True
        except Exception as e:
            logger.error(f"Failed to record alert: {e}")
            return False
        finally:
            conn.close()

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent budget alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert records
        """
        conn = self._get_connection()
        try:
            results = conn.execute("""
                SELECT *
                FROM budget_alerts
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get alerts: {e}")
            return []
        finally:
            conn.close()

    def clear_old_records(self, days: int = 90) -> int:
        """
        Clear cost records older than specified days.

        Args:
            days: Keep records newer than this many days

        Returns:
            Number of records deleted
        """
        conn = self._get_connection()
        try:
            cutoff_time = time.time() - (days * 86400)
            cursor = conn.execute("""
                DELETE FROM cost_records
                WHERE timestamp < ?
            """, (cutoff_time,))
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleared {deleted} old cost records")
            return deleted
        except Exception as e:
            logger.error(f"Failed to clear old records: {e}")
            return 0
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get tracker statistics.

        Returns:
            Dictionary with tracker stats
        """
        conn = self._get_connection()
        try:
            # Get total records
            total = conn.execute("SELECT COUNT(*) as count FROM cost_records").fetchone()

            # Get date range
            date_range = conn.execute("""
                SELECT
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM cost_records
            """).fetchone()

            # Get total cost
            total_cost = conn.execute("""
                SELECT SUM(cost_usd) as total FROM cost_records
            """).fetchone()

            return {
                "total_records": total["count"] or 0,
                "first_record": date_range["first_record"],
                "last_record": date_range["last_record"],
                "total_cost_tracked": total_cost["total"] or 0.0,
                "db_path": self.db_path,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
        finally:
            conn.close()

    def close(self) -> None:
        """Clean up resources."""
        # SQLite connections are closed after each operation
        logger.info("Budget tracker closed")


# Global singleton
_tracker: Optional[BudgetTracker] = None


def get_budget_tracker(db_path: str = "./data/budget.db") -> BudgetTracker:
    """
    Get global budget tracker instance.

    Args:
        db_path: Database path

    Returns:
        BudgetTracker instance
    """
    global _tracker

    if _tracker is None:
        _tracker = BudgetTracker(db_path)

    return _tracker


def close_budget_tracker() -> None:
    """Close global budget tracker."""
    global _tracker

    if _tracker:
        _tracker.close()
        _tracker = None
