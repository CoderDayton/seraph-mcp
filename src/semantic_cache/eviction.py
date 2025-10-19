"""
Semantic Cache Eviction - Multi-Layer LRU+FIFO

Implements production-grade cache eviction with multi-tier architecture:
- Hot tier: Small LRU cache for frequently accessed items
- Cold tier: Large FIFO cache as second-level buffer
- Promotion: Items in FIFO promoted to LRU on re-access before eviction
- Optional TTL: Time-based expiration (disabled by default)

Per SDD.md P0 Phase 3:
- Uses cachetools for O(1) operations
- TTL disabled by default (TTL=0)
- Tracks hits, misses, evictions, promotions
- ChromaDB batch deletes for efficiency
- Target: <5ms overhead

Architecture:
    Query → LRU (hot) → hit ✓
              ↓ miss
           FIFO (cold) → hit → promote to LRU
              ↓ miss
          ChromaDB lookup
"""

import logging
import time
from collections import deque
from typing import Any

# Try to import cachetools implementations
try:
    from cachetools import FIFOCache, LRUCache, TTLCache

    CACHETOOLS_AVAILABLE = True
except ImportError:  # pragma: no cover
    # Mark cachetools as unavailable
    CACHETOOLS_AVAILABLE = False

    class LRUCache:  # type: ignore[no-redef]
        """Fallback LRU implementation when cachetools is not available."""

        def __init__(self, maxsize: int, **kwargs: Any) -> None:
            self.maxsize = maxsize
            self._data: dict[Any, Any] = {}

        def __contains__(self, key: Any) -> bool:
            return key in self._data

        def __getitem__(self, key: Any) -> Any:
            return self._data[key]

        def __setitem__(self, key: Any, value: Any) -> None:
            self._data[key] = value

        def __delitem__(self, key: Any) -> None:
            del self._data[key]

        def clear(self) -> None:
            self._data.clear()

        def __len__(self) -> int:
            return len(self._data)

    class FIFOCache:  # type: ignore[no-redef]
        """Fallback FIFO implementation when cachetools is not available."""

        def __init__(self, maxsize: int, **kwargs: Any) -> None:
            self.maxsize = maxsize
            self._data: dict[Any, Any] = {}

        def __contains__(self, key: Any) -> bool:
            return key in self._data

        def __getitem__(self, key: Any) -> Any:
            return self._data[key]

        def __setitem__(self, key: Any, value: Any) -> None:
            self._data[key] = value

        def __delitem__(self, key: Any) -> None:
            del self._data[key]

        def clear(self) -> None:
            self._data.clear()

        def __len__(self) -> int:
            return len(self._data)

    class TTLCache:  # type: ignore[no-redef]
        """Fallback TTL implementation when cachetools is not available."""

        def __init__(self, maxsize: int, ttl: int = 0, **kwargs: Any) -> None:
            self.maxsize = maxsize
            self.ttl = ttl
            self._data: dict[Any, Any] = {}

        def __contains__(self, key: Any) -> bool:
            return key in self._data

        def __getitem__(self, key: Any) -> Any:
            return self._data[key]

        def __setitem__(self, key: Any, value: Any) -> None:
            self._data[key] = value

        def __delitem__(self, key: Any) -> None:
            del self._data[key]

        def clear(self) -> None:
            self._data.clear()

        def __len__(self) -> int:
            return len(self._data)


logger = logging.getLogger(__name__)


class EvictionStats:
    """Statistics tracker for eviction system."""

    def __init__(self) -> None:
        """Initialize stats counters."""
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.promotions: int = 0
        self.ttl_expired: int = 0
        self.lru_hits: int = 0
        self.fifo_hits: int = 0
        self.chromadb_hits: int = 0

    def reset(self) -> None:
        """Reset all counters."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.promotions = 0
        self.ttl_expired = 0
        self.lru_hits = 0
        self.fifo_hits = 0
        self.chromadb_hits = 0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "promotions": self.promotions,
            "ttl_expired": self.ttl_expired,
            "hit_rate": self.get_hit_rate(),
            "lru_hits": self.lru_hits,
            "fifo_hits": self.fifo_hits,
            "chromadb_hits": self.chromadb_hits,
        }


class MultiLayerCache:
    """
    Multi-layer cache with LRU (hot) + FIFO (cold) eviction.

    Architecture:
        LRU (hot tier): Small, fast cache for most recently accessed items
        FIFO (cold tier): Larger buffer for less frequent items
        Promotion: FIFO items promoted to LRU on access before eviction

    Config:
        lru_size: Size of hot LRU tier (default: 100)
        fifo_size: Size of cold FIFO tier (default: 900)
        ttl_seconds: Time-to-live for entries (0 = disabled, default)
        high_watermark_pct: Trigger cleanup at this % (default: 90)
        cleanup_batch_size: Entries to evict per cleanup (default: 100)
    """

    def __init__(
        self,
        lru_size: int = 100,
        fifo_size: int = 900,
        ttl_seconds: int = 0,
        high_watermark_pct: int = 90,
        cleanup_batch_size: int = 100,
    ):
        """
        Initialize multi-layer cache.

        Args:
            lru_size: Size of hot LRU tier
            fifo_size: Size of cold FIFO tier
            ttl_seconds: Time-to-live in seconds (0 = disabled)
            high_watermark_pct: Percentage to trigger cleanup
            cleanup_batch_size: Number of entries to evict per cleanup
        """
        self.lru_size = lru_size
        self.fifo_size = fifo_size
        self.ttl_seconds = ttl_seconds
        self.high_watermark_pct = high_watermark_pct
        self.cleanup_batch_size = cleanup_batch_size

        # Initialize caches - use Any for type since we support both cachetools and fallback
        self._lru: Any
        self._fifo: Any

        if ttl_seconds > 0:
            # Use TTL cache for both tiers
            self._lru = TTLCache(maxsize=lru_size, ttl=ttl_seconds)
            self._fifo = TTLCache(maxsize=fifo_size, ttl=ttl_seconds)
        else:
            # Use LRU + FIFO without TTL
            self._lru = LRUCache(maxsize=lru_size)
            self._fifo = FIFOCache(maxsize=fifo_size)

        if not CACHETOOLS_AVAILABLE:
            logger.warning("cachetools not available, using fallback implementations (limited functionality)")

        # Eviction queue for batch ChromaDB deletes
        self._eviction_queue: deque[str] = deque(maxlen=cleanup_batch_size)

        # Metadata tracking
        self._entry_timestamps: dict[str, float] = {}
        self._entry_metadata: dict[str, dict[str, Any]] = {}

        # Statistics
        self._stats = EvictionStats()

        logger.info(
            f"Multi-layer cache initialized: LRU={lru_size}, FIFO={fifo_size}, "
            f"TTL={ttl_seconds}s, watermark={high_watermark_pct}%"
        )

    def get(self, key: str) -> dict[str, Any] | None:
        """
        Get value from cache with multi-tier lookup.

        Lookup order:
            1. LRU (hot tier) → hit
            2. FIFO (cold tier) → hit → promote to LRU
            3. Miss

        Args:
            key: Cache key

        Returns:
            Cached entry dict or None
        """
        # Try LRU first (hot tier)
        if key in self._lru:
            self._stats.hits += 1
            self._stats.lru_hits += 1
            lru_entry: dict[str, Any] = self._lru[key]
            return lru_entry

        # Try FIFO (cold tier)
        if key in self._fifo:
            self._stats.hits += 1
            self._stats.fifo_hits += 1

            # Promote to LRU before returning
            fifo_entry: dict[str, Any] = self._fifo[key]
            self._promote_to_lru(key, fifo_entry)
            return fifo_entry

        # Cache miss
        self._stats.misses += 1
        return None

    def set(
        self,
        key: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store value in cache (always goes to LRU hot tier).

        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional metadata
        """
        now = time.time()

        # Store in LRU (hot tier)
        self._lru[key] = value

        # Track metadata
        self._entry_timestamps[key] = now
        if metadata:
            self._entry_metadata[key] = metadata

        # Check if we need cleanup
        self._check_watermark()

    def _promote_to_lru(self, key: str, entry: Any) -> None:
        """
        Promote entry from FIFO to LRU.

        Args:
            key: Cache key
            entry: Cached value
        """
        # Move to LRU
        self._lru[key] = entry
        self._stats.promotions += 1

        # Remove from FIFO
        if key in self._fifo:
            del self._fifo[key]

    def _check_watermark(self) -> None:
        """Check if we've exceeded high watermark and trigger cleanup."""
        total_entries = len(self._lru) + len(self._fifo)
        capacity = self.lru_size + self.fifo_size
        utilization_pct = (total_entries / capacity) * 100

        if utilization_pct >= self.high_watermark_pct:
            logger.warning(f"Cache at {utilization_pct:.1f}% capacity, triggering cleanup")
            self._cleanup_old_entries()

    def _cleanup_old_entries(self) -> None:
        """
        Clean up old entries based on age.

        Removes oldest entries from FIFO tier first, then LRU if needed.
        Adds evicted keys to batch delete queue for ChromaDB.
        """
        target_evictions = min(self.cleanup_batch_size, len(self._fifo) + len(self._lru))

        if target_evictions == 0:
            return

        evicted_keys: list[str] = []

        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(self._entry_timestamps.items(), key=lambda x: x[1])

        # Evict oldest entries
        for key, _ in sorted_entries[:target_evictions]:
            # Remove from caches
            if key in self._fifo:
                del self._fifo[key]
            elif key in self._lru:
                del self._lru[key]

            # Clean metadata
            self._entry_timestamps.pop(key, None)
            self._entry_metadata.pop(key, None)

            # Add to eviction queue
            evicted_keys.append(key)
            self._stats.evictions += 1

        # Add to batch delete queue
        for key in evicted_keys:
            self._eviction_queue.append(key)

        logger.info(f"Evicted {len(evicted_keys)} entries from cache")

    def get_eviction_queue(self) -> list[str]:
        """
        Get and clear eviction queue for batch ChromaDB delete.

        Returns:
            List of keys to delete from ChromaDB
        """
        keys = list(self._eviction_queue)
        self._eviction_queue.clear()
        return keys

    def clear(self) -> None:
        """Clear all cache tiers."""
        self._lru.clear()
        self._fifo.clear()
        self._entry_timestamps.clear()
        self._entry_metadata.clear()
        self._eviction_queue.clear()
        logger.info("Multi-layer cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary with utilization, hit rates, etc.
        """
        total_entries = len(self._lru) + len(self._fifo)
        capacity = self.lru_size + self.fifo_size
        utilization_pct = (total_entries / capacity) * 100 if capacity > 0 else 0.0

        # Calculate oldest entry age
        oldest_age_sec = 0.0
        if self._entry_timestamps:
            oldest_timestamp = min(self._entry_timestamps.values())
            oldest_age_sec = time.time() - oldest_timestamp

        stats_dict = self._stats.to_dict()
        stats_dict.update(
            {
                "lru_entries": len(self._lru),
                "fifo_entries": len(self._fifo),
                "total_entries": total_entries,
                "capacity": capacity,
                "utilization_pct": utilization_pct,
                "oldest_entry_age_sec": oldest_age_sec,
                "eviction_queue_size": len(self._eviction_queue),
                "config": {
                    "lru_size": self.lru_size,
                    "fifo_size": self.fifo_size,
                    "ttl_seconds": self.ttl_seconds,
                    "high_watermark_pct": self.high_watermark_pct,
                    "cleanup_batch_size": self.cleanup_batch_size,
                },
            }
        )

        return stats_dict

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats.reset()
