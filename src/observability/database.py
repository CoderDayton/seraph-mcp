"""
Seraph MCP — Metrics Database Manager

Handles SQLite database connection, initialization, and async session management.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .db_models import Base


class MetricsDatabase:
    """
    Async SQLite database manager for metrics storage.

    Provides:
    - Automatic schema creation/migration
    - Async session management
    - Connection pooling
    - Graceful shutdown
    """

    def __init__(self, db_path: str = "./data/metrics.db"):
        """
        Initialize metrics database.

        Args:
            db_path: Path to SQLite database file (relative or absolute)
        """
        self.db_path = Path(db_path).resolve()
        self.db_url = f"sqlite+aiosqlite:///{self.db_path}"

        # Create async engine with connection pooling
        self.engine: AsyncEngine = create_async_engine(
            self.db_url,
            echo=False,  # Set to True for SQL query logging
            connect_args={"check_same_thread": False},  # Required for SQLite
        )

        # Create async session factory
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        self._initialized = False
        self._initialization_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """
        Initialize database schema.

        Creates all tables if they don't exist.
        Safe to call multiple times (idempotent).
        """
        async with self._initialization_lock:
            if self._initialized:
                return

            # Create parent directory if needed
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create all tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session (context manager).

        Usage:
            async with db.get_session() as session:
                await session.execute(...)
                await session.commit()
        """
        if not self._initialized:
            await self.initialize()

        async with self.session_factory() as session:
            yield session

    async def close(self) -> None:
        """Close database connections gracefully."""
        await self.engine.dispose()
        self._initialized = False
