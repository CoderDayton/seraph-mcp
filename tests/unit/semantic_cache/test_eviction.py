"""
Tests for Multi-Layer Cache Eviction System

Comprehensive test coverage for LRU+FIFO eviction with promotion logic.

P0 Phase 3 Testing:
- LRU hot tier behavior
- FIFO cold tier behavior
- Promotion from FIFO to LRU
- High watermark cleanup
- TTL expiration
- Eviction queue for batch deletes
- Statistics tracking
"""

import time

from src.semantic_cache.eviction import EvictionStats, MultiLayerCache


class TestEvictionStats:
    """Test EvictionStats tracker."""

    def test_init(self):
        """Test stats initialization."""
        stats = EvictionStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.promotions == 0
        assert stats.ttl_expired == 0
        assert stats.lru_hits == 0
        assert stats.fifo_hits == 0
        assert stats.chromadb_hits == 0

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        stats = EvictionStats()

        # No hits or misses = 0.0
        assert stats.get_hit_rate() == 0.0

        # 3 hits, 1 miss = 0.75
        stats.hits = 3
        stats.misses = 1
        assert stats.get_hit_rate() == 0.75

        # 10 hits, 0 misses = 1.0
        stats.hits = 10
        stats.misses = 0
        assert stats.get_hit_rate() == 1.0

    def test_reset(self):
        """Test stats reset."""
        stats = EvictionStats()
        stats.hits = 10
        stats.misses = 5
        stats.evictions = 3
        stats.promotions = 2

        stats.reset()

        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.promotions == 0

    def test_to_dict(self):
        """Test stats dictionary export."""
        stats = EvictionStats()
        stats.hits = 8
        stats.misses = 2
        stats.lru_hits = 5
        stats.fifo_hits = 3

        result = stats.to_dict()

        assert result["hits"] == 8
        assert result["misses"] == 2
        assert result["hit_rate"] == 0.8
        assert result["lru_hits"] == 5
        assert result["fifo_hits"] == 3


class TestMultiLayerCache:
    """Test MultiLayerCache with LRU+FIFO architecture."""

    def test_init_defaults(self):
        """Test cache initialization with defaults."""
        cache = MultiLayerCache()

        assert cache.lru_size == 100
        assert cache.fifo_size == 900
        assert cache.ttl_seconds == 0
        assert cache.high_watermark_pct == 90
        assert cache.cleanup_batch_size == 100

    def test_init_custom(self):
        """Test cache initialization with custom config."""
        cache = MultiLayerCache(
            lru_size=50,
            fifo_size=450,
            ttl_seconds=60,
            high_watermark_pct=85,
            cleanup_batch_size=50,
        )

        assert cache.lru_size == 50
        assert cache.fifo_size == 450
        assert cache.ttl_seconds == 60
        assert cache.high_watermark_pct == 85
        assert cache.cleanup_batch_size == 50

    def test_lru_hit(self):
        """Test LRU hot tier cache hit."""
        cache = MultiLayerCache(lru_size=10, fifo_size=10)

        # Store in cache
        cache.set("key1", "value1")

        # Retrieve from LRU (hot tier)
        result = cache.get("key1")

        assert result == "value1"
        stats = cache.get_stats()
        assert stats["lru_hits"] == 1
        assert stats["fifo_hits"] == 0
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_fifo_hit_with_promotion(self):
        """Test FIFO cold tier hit promotes to LRU."""
        cache = MultiLayerCache(lru_size=2, fifo_size=5)

        # Fill LRU
        cache.set("lru1", "value1")
        cache.set("lru2", "value2")

        # Add to FIFO (LRU is full, so it goes to FIFO)
        cache._fifo["fifo1"] = "value_fifo"
        cache._entry_timestamps["fifo1"] = time.time()

        # Retrieve from FIFO - should promote to LRU
        result = cache.get("fifo1")

        assert result == "value_fifo"
        stats = cache.get_stats()
        assert stats["fifo_hits"] == 1
        assert stats["promotions"] == 1
        # LRU is capped at size 2, so promoting evicts oldest LRU item
        assert stats["lru_entries"] == 2  # Still 2 (auto-evicted oldest)
        assert stats["fifo_entries"] == 0  # Removed from FIFO

    def test_cache_miss(self):
        """Test cache miss tracking."""
        cache = MultiLayerCache(lru_size=10, fifo_size=10)

        # Try to get non-existent key
        result = cache.get("nonexistent")

        assert result is None
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0

    def test_metadata_tracking(self):
        """Test metadata is tracked with cache entries."""
        cache = MultiLayerCache(lru_size=10, fifo_size=10)

        metadata = {"source": "test", "priority": 5}
        cache.set("key1", "value1", metadata=metadata)

        # Verify metadata is stored
        assert "key1" in cache._entry_metadata
        assert cache._entry_metadata["key1"] == metadata

    def test_high_watermark_cleanup(self):
        """Test automatic cleanup at high watermark."""
        cache = MultiLayerCache(
            lru_size=5,
            fifo_size=5,
            high_watermark_pct=80,  # 80% of 10 = 8 entries
            cleanup_batch_size=3,
        )

        # Fill LRU first (5 entries)
        for i in range(5):
            cache.set(f"lru{i}", f"value{i}")
            time.sleep(0.001)

        # LRU is full, so next entries go to FIFO
        # Add 2 to FIFO (total 7 entries = 70%)
        for i in range(2):
            cache._fifo[f"fifo{i}"] = f"value_fifo{i}"
            cache._entry_timestamps[f"fifo{i}"] = time.time()
            time.sleep(0.001)

        stats = cache.get_stats()
        assert stats["total_entries"] == 7
        assert stats["evictions"] == 0

        # Add 2 more to FIFO to trigger cleanup (90% > 80% watermark)
        cache._fifo["fifo2"] = "value_fifo2"
        cache._entry_timestamps["fifo2"] = time.time()
        time.sleep(0.001)
        cache._fifo["fifo3"] = "value_fifo3"
        cache._entry_timestamps["fifo3"] = time.time()

        # Manually trigger watermark check
        cache._check_watermark()

        # Cleanup should have been triggered
        stats = cache.get_stats()
        assert stats["evictions"] > 0

    def test_eviction_queue(self):
        """Test eviction queue for batch ChromaDB deletes."""
        cache = MultiLayerCache(
            lru_size=3,
            fifo_size=3,
            high_watermark_pct=80,  # 80% of 6 = 4.8, so 5+ triggers
            cleanup_batch_size=2,
        )

        # Fill LRU first (3 entries)
        for i in range(3):
            cache.set(f"lru{i}", f"value{i}")
            time.sleep(0.001)

        # Add to FIFO (3 more entries, total = 6)
        for i in range(3):
            cache._fifo[f"fifo{i}"] = f"value_fifo{i}"
            cache._entry_timestamps[f"fifo{i}"] = time.time()
            time.sleep(0.001)

        # Manually trigger cleanup by checking watermark
        cache._check_watermark()

        # Get eviction queue
        evicted_keys = cache.get_eviction_queue()

        # Should have keys in queue after cleanup
        assert len(evicted_keys) > 0

        # Queue should be cleared after retrieval
        evicted_keys_2 = cache.get_eviction_queue()
        assert len(evicted_keys_2) == 0

    def test_clear(self):
        """Test cache clear operation."""
        cache = MultiLayerCache(lru_size=10, fifo_size=10)

        # Add entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache._fifo["key3"] = "value3"

        # Clear
        cache.clear()

        stats = cache.get_stats()
        assert stats["lru_entries"] == 0
        assert stats["fifo_entries"] == 0
        assert stats["total_entries"] == 0

    def test_statistics_comprehensive(self):
        """Test comprehensive statistics reporting."""
        cache = MultiLayerCache(lru_size=5, fifo_size=5)

        # Add entries
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Hits and misses
        cache.get("key1")  # LRU hit
        cache.get("missing")  # Miss

        stats = cache.get_stats()

        assert "lru_entries" in stats
        assert "fifo_entries" in stats
        assert "total_entries" in stats
        assert "capacity" in stats
        assert "utilization_pct" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "evictions" in stats
        assert "promotions" in stats
        assert "oldest_entry_age_sec" in stats
        assert "eviction_queue_size" in stats
        assert "config" in stats

        # Verify calculated values
        assert stats["capacity"] == 10
        assert stats["total_entries"] == 2
        assert stats["utilization_pct"] == 20.0
        assert stats["hit_rate"] == 0.5  # 1 hit, 1 miss

    def test_oldest_entry_age(self):
        """Test oldest entry age tracking."""
        cache = MultiLayerCache(lru_size=10, fifo_size=10)

        # Add entry
        cache.set("key1", "value1")
        time.sleep(0.1)

        stats = cache.get_stats()

        # Should have age > 0
        assert stats["oldest_entry_age_sec"] >= 0.1
        assert stats["oldest_entry_age_sec"] < 1.0

    def test_reset_stats(self):
        """Test statistics reset."""
        cache = MultiLayerCache(lru_size=10, fifo_size=10)

        # Generate some stats
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("missing")

        stats_before = cache.get_stats()
        assert stats_before["hits"] > 0

        # Reset
        cache.reset_stats()

        stats_after = cache.get_stats()
        assert stats_after["hits"] == 0
        assert stats_after["misses"] == 0

    def test_ttl_cache_initialization(self):
        """Test TTL-enabled cache initialization."""
        cache = MultiLayerCache(
            lru_size=10,
            fifo_size=10,
            ttl_seconds=60,
        )

        # Should use TTL cache
        assert cache.ttl_seconds == 60

    def test_multiple_promotions(self):
        """Test multiple items can be promoted."""
        cache = MultiLayerCache(lru_size=3, fifo_size=5)

        # Fill LRU
        cache.set("lru1", "v1")
        cache.set("lru2", "v2")
        cache.set("lru3", "v3")

        # Add to FIFO
        cache._fifo["fifo1"] = "vf1"
        cache._fifo["fifo2"] = "vf2"
        cache._entry_timestamps["fifo1"] = time.time()
        cache._entry_timestamps["fifo2"] = time.time()

        # Promote both
        cache.get("fifo1")
        cache.get("fifo2")

        stats = cache.get_stats()
        assert stats["promotions"] == 2
        assert stats["fifo_entries"] == 0

    def test_lru_eviction_on_overflow(self):
        """Test LRU evicts least recently used when full."""
        cache = MultiLayerCache(lru_size=3, fifo_size=3)

        # Fill LRU
        cache.set("key1", "v1")
        cache.set("key2", "v2")
        cache.set("key3", "v3")

        # Access key1 to make it recently used
        cache.get("key1")

        # Add more items - should evict key2 or key3 (not key1)
        cache.set("key4", "v4")

        # key1 should still be in LRU (most recently accessed)
        result = cache.get("key1")
        assert result == "v1"
        stats = cache.get_stats()
        assert stats["lru_hits"] >= 2  # Initial get + verification get

    def test_config_in_stats(self):
        """Test configuration is included in stats."""
        cache = MultiLayerCache(
            lru_size=100,
            fifo_size=900,
            ttl_seconds=3600,
            high_watermark_pct=85,
            cleanup_batch_size=50,
        )

        stats = cache.get_stats()
        config = stats["config"]

        assert config["lru_size"] == 100
        assert config["fifo_size"] == 900
        assert config["ttl_seconds"] == 3600
        assert config["high_watermark_pct"] == 85
        assert config["cleanup_batch_size"] == 50
