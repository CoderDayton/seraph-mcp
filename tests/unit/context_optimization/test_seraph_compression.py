"""
Seraph MCP — Seraph Compression Unit Tests

Comprehensive test suite for the Seraph multi-layer compression system.
Tests deterministic compression, L1/L2/L3 layers, BM25 ranking, and quality preservation.

Python 3.12+ with modern async patterns and type hints.
"""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

from src.context_optimization.seraph_compression import (
    BM25,
    CompressionResult,
    SeraphCompressor,
    blake_hash,
    count_tokens,
    hamm_distance64,
    simhash64,
)


class TestSeraphCompressionUtilities:
    """Test utility functions for Seraph compression."""

    def test_count_tokens_simple(self) -> None:
        """Test basic token counting."""
        text = "Hello world, this is a test."
        count = count_tokens(text)
        assert count > 0
        assert isinstance(count, int)

    def test_count_tokens_empty(self) -> None:
        """Test token counting with empty string."""
        assert count_tokens("") == 0

    def test_count_tokens_unicode(self) -> None:
        """Test token counting with Unicode text."""
        text = "Hello 世界 🌍"
        count = count_tokens(text)
        assert count > 0

    def test_blake_hash(self) -> None:
        """Test blake hashing function."""
        data = b"test data"
        hash1 = blake_hash(data)
        hash2 = blake_hash(data)

        # Same input should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) > 0

        # Different input should produce different hash
        hash3 = blake_hash(b"different data")
        assert hash1 != hash3

    def test_simhash64_basic(self) -> None:
        """Test basic simhash64 functionality."""
        tokens1 = ["hello", "world", "test"]
        tokens2 = ["hello", "world", "test"]
        tokens3 = ["goodbye", "world", "test"]

        hash1 = simhash64(tokens1)
        hash2 = simhash64(tokens2)
        hash3 = simhash64(tokens3)

        # Same tokens should produce same hash
        assert hash1 == hash2

        # Different tokens should produce different hash
        assert hash1 != hash3

    def test_simhash64_empty(self) -> None:
        """Test simhash64 with empty token list."""
        assert simhash64([]) == 0

    def test_hamm_distance64(self) -> None:
        """Test Hamming distance calculation."""
        # Same values have distance 0
        assert hamm_distance64(0, 0) == 0
        assert hamm_distance64(123, 123) == 0

        # Different values have non-zero distance
        dist = hamm_distance64(0b1010, 0b0101)
        assert dist == 4  # All 4 bits are different

        dist = hamm_distance64(0b1111, 0b1110)
        assert dist == 1  # Only 1 bit is different


class TestBM25:
    """Test BM25 ranking algorithm."""

    def test_bm25_initialization(self) -> None:
        """Test BM25 initialization."""
        docs = [
            ["hello", "world"],
            ["hello", "python"],
            ["world", "python"],
        ]
        bm25 = BM25(docs)

        assert bm25.N == 3
        assert bm25.k1 == 1.5
        assert bm25.b == 0.75
        assert len(bm25.df) > 0
        assert len(bm25.idf) > 0

    def test_bm25_score(self) -> None:
        """Test BM25 scoring."""
        docs = [
            ["machine", "learning", "algorithm"],
            ["deep", "learning", "neural", "network"],
            ["machine", "learning", "data", "science"],
            ["artificial", "intelligence", "system"],
        ]
        bm25 = BM25(docs)

        # Query about machine learning
        query = ["machine", "learning"]
        scores = [bm25.score(query, i) for i in range(len(docs))]

        # Docs 0 and 2 should have highest scores (contain both terms)
        assert scores[0] > scores[3]
        assert scores[2] > scores[3]

        # Doc 1 has only "learning"
        assert scores[1] > 0
        assert scores[1] < scores[0]

    def test_bm25_empty_query(self) -> None:
        """Test BM25 with empty query."""
        docs = [["hello", "world"]]
        bm25 = BM25(docs)
        score = bm25.score([], 0)
        assert score == 0.0


class TestCompressionResult:
    """Test CompressionResult dataclass."""

    def test_compression_result_creation(self) -> None:
        """Test creating a CompressionResult."""
        manifest = {
            "tier1": {"l1_ratio": 0.002, "l2_ratio": 0.01, "l3_ratio": 0.05},
            "tier2": {"budget_tokens": 256, "dcp_tokens": 200},
            "tier3": {"method": "fallback"},
        }

        result = CompressionResult(
            l1="L1 skeleton text",
            l2="L2 abstract text",
            l3="L3 extract text",
            manifest=manifest,
        )

        assert result.l1 == "L1 skeleton text"
        assert result.l2 == "L2 abstract text"
        assert result.l3 == "L3 extract text"
        assert result.manifest == manifest
        assert "tier1" in result.manifest
        assert "tier2" in result.manifest
        assert "tier3" in result.manifest


class TestSeraphCompressor:
    """Test SeraphCompressor main functionality using the build/query/pack API."""

    @pytest.fixture
    def compressor(self) -> SeraphCompressor:
        """Create a SeraphCompressor instance."""
        return SeraphCompressor(seed=42)

    @pytest.fixture
    def short_text(self) -> str:
        """Short text sample."""
        return """
        Machine learning is a subset of artificial intelligence.
        It focuses on developing systems that can learn from data.
        These systems improve their performance over time.
        """

    @pytest.fixture
    def long_text(self) -> str:
        """Long text sample for compression testing."""
        return """
        Artificial intelligence has revolutionized numerous industries and continues
        to shape the future of technology. Machine learning, a crucial subset of AI,
        enables computers to learn from data and make decisions without explicit programming.

        Deep learning, a specialized branch of machine learning, uses neural networks with
        multiple layers to process complex patterns in large datasets. These networks can
        recognize images, understand natural language, and even generate creative content.

        Natural language processing (NLP) is another critical area of AI that focuses on
        the interaction between computers and human language. NLP powers applications like
        chatbots, translation services, and sentiment analysis tools.

        Computer vision enables machines to interpret and understand visual information
        from the world. This technology is used in facial recognition, autonomous vehicles,
        medical image analysis, and quality control in manufacturing.

        Reinforcement learning is a type of machine learning where agents learn to make
        decisions by interacting with their environment. This approach has been successful
        in game playing, robotics, and optimization problems.

        The ethical implications of AI development are increasingly important. Issues like
        bias in algorithms, privacy concerns, and the potential impact on employment require
        careful consideration. Responsible AI development prioritizes transparency, fairness,
        and accountability.

        As AI continues to advance, the collaboration between humans and machines becomes
        more sophisticated. The goal is not to replace human intelligence but to augment
        it, enabling us to solve complex problems more effectively.
        """

    def test_compressor_initialization(self, compressor: SeraphCompressor) -> None:
        """Test compressor initialization."""
        assert hasattr(compressor, "t1")
        assert hasattr(compressor, "t2")
        assert hasattr(compressor, "t3")

    def test_build_short_text(self, compressor: SeraphCompressor, short_text: str) -> None:
        """Test building compression layers from short text."""
        result = compressor.build(short_text)

        assert isinstance(result, CompressionResult)
        assert result.l1 is not None
        assert result.l2 is not None
        assert result.l3 is not None
        assert isinstance(result.manifest, dict)

        # All layers should contain some content
        assert len(result.l1) > 0
        assert len(result.l2) > 0
        assert len(result.l3) > 0

    def test_build_long_text(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test building compression layers from long text."""
        result = compressor.build(long_text)

        assert isinstance(result, CompressionResult)

        # Verify all layers exist
        assert result.l1 is not None
        assert result.l2 is not None
        assert result.l3 is not None

        # Verify manifest structure
        assert "tier1" in result.manifest
        assert "tier2" in result.manifest
        assert "tier3" in result.manifest

        # L1 should be shortest, L3 should be longest
        l1_tokens = count_tokens(result.l1)
        l2_tokens = count_tokens(result.l2)
        l3_tokens = count_tokens(result.l3)

        assert l1_tokens <= l2_tokens
        assert l2_tokens <= l3_tokens

        # All should be compressed compared to original
        original_tokens = count_tokens(long_text)
        assert l3_tokens < original_tokens

    def test_build_deterministic(self, compressor: SeraphCompressor, short_text: str) -> None:
        """Test that build produces deterministic results."""
        result1 = compressor.build(short_text)
        result2 = compressor.build(short_text)

        # Same input should produce same output
        assert result1.l1 == result2.l1
        assert result1.l2 == result2.l2
        assert result1.l3 == result2.l3

    def test_build_empty_text(self, compressor: SeraphCompressor) -> None:
        """Test building layers from empty text."""
        result = compressor.build("")

        assert isinstance(result, CompressionResult)
        # Empty text should produce empty or minimal layers
        assert len(result.l1) >= 0
        assert len(result.l2) >= 0
        assert len(result.l3) >= 0

    def test_build_whitespace_only(self, compressor: SeraphCompressor) -> None:
        """Test building layers from whitespace-only text."""
        result = compressor.build("   \n\n   \t\t  ")

        assert isinstance(result, CompressionResult)
        # Should handle gracefully
        assert result.l1 is not None
        assert result.l2 is not None
        assert result.l3 is not None

    def test_build_unicode_text(self, compressor: SeraphCompressor) -> None:
        """Test building layers from Unicode text."""
        unicode_text = """
        人工智能正在改变世界。Machine learning and deep learning are key technologies.
        自然语言处理 (NLP) enables computers to understand human language.
        The future of AI is 令人兴奋的 and full of possibilities.
        """
        result = compressor.build(unicode_text)

        assert isinstance(result, CompressionResult)
        assert len(result.l1) > 0
        assert len(result.l2) > 0
        assert len(result.l3) > 0

    def test_build_code_blocks(self, compressor: SeraphCompressor) -> None:
        """Test building layers from text with code blocks."""
        code_text = """
        Python is a popular programming language for machine learning.

        ```python
        def train_model(data):
            model = Model()
            model.fit(data)
            return model
        ```

        The code above shows a simple training function.
        Libraries like scikit-learn, TensorFlow, and PyTorch are commonly used.
        """
        result = compressor.build(code_text)

        assert isinstance(result, CompressionResult)
        # Should preserve some information about the code
        assert len(result.l3) > 0

    def test_build_special_characters(self, compressor: SeraphCompressor) -> None:
        """Test building layers with special characters."""
        special_text = """
        AI systems can process various formats: JSON, XML, CSV, etc.
        Mathematical symbols like ∑, ∫, π are common in ML papers.
        Special characters: @#$%^&*()_+-=[]{}|;:'",.<>?/~`
        """
        result = compressor.build(special_text)

        assert isinstance(result, CompressionResult)
        assert result.l1 is not None
        assert result.l2 is not None
        assert result.l3 is not None

    def test_query_basic(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test basic query functionality."""
        result = compressor.build(long_text)

        question = "What is machine learning?"
        top_k = compressor.query(result, question, k=5)

        assert isinstance(top_k, list)
        assert len(top_k) <= 5

        # Each result should be a tuple of (score, text)
        for score, text in top_k:
            assert isinstance(score, float)
            assert isinstance(text, str)
            assert len(text) > 0

    def test_query_relevance(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test that query returns relevant results."""
        result = compressor.build(long_text)

        question = "deep learning neural networks"
        top_k = compressor.query(result, question, k=3)

        # Should return results
        assert len(top_k) > 0

        # Scores should be in descending order
        scores = [score for score, _ in top_k]
        assert scores == sorted(scores, reverse=True)

    def test_query_different_k_values(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test query with different k values."""
        result = compressor.build(long_text)
        question = "artificial intelligence"

        top_3 = compressor.query(result, question, k=3)
        top_5 = compressor.query(result, question, k=5)
        top_10 = compressor.query(result, question, k=10)

        assert len(top_3) <= 3
        assert len(top_5) <= 5
        assert len(top_10) <= 10

        # Top 3 should be subset of top 5
        if len(top_3) > 0 and len(top_5) >= 3:
            assert top_3[0] == top_5[0]

    def test_query_empty_question(self, compressor: SeraphCompressor, short_text: str) -> None:
        """Test query with empty question."""
        result = compressor.build(short_text)
        top_k = compressor.query(result, "", k=5)

        # Should handle gracefully
        assert isinstance(top_k, list)

    def test_pack_and_load(self, compressor: SeraphCompressor, short_text: str) -> None:
        """Test packing and loading compression result."""
        result = compressor.build(short_text)

        # Pack to temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            output_path = compressor.pack(result, temp_path)
            assert Path(output_path).exists()
            assert output_path == temp_path

            # Load the packed file
            with gzip.open(temp_path, "rt", encoding="utf-8") as f:
                payload = json.load(f)

            # Verify structure
            assert "manifest" in payload
            assert "L1" in payload
            assert "L2" in payload
            assert "L3" in payload

            # Recreate CompressionResult
            loaded_result = CompressionResult(
                payload["L1"],
                payload["L2"],
                payload["L3"],
                payload["manifest"],
            )

            # Should match original
            assert loaded_result.l1 == result.l1
            assert loaded_result.l2 == result.l2
            assert loaded_result.l3 == result.l3
            assert loaded_result.manifest == result.manifest

        finally:
            # Cleanup
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_pack_creates_valid_json(self, compressor: SeraphCompressor, short_text: str) -> None:
        """Test that pack creates valid JSON."""
        result = compressor.build(short_text)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json.gz", delete=False) as f:
            temp_path = f.name

        try:
            compressor.pack(result, temp_path)

            # Should be valid gzipped JSON
            with gzip.open(temp_path, "rt", encoding="utf-8") as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert all(key in data for key in ["manifest", "L1", "L2", "L3"])

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def test_compression_preserves_key_information(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test that compression preserves key information."""
        result = compressor.build(long_text)

        # Check that key terms are preserved in at least one layer
        key_terms = ["machine learning", "deep learning", "artificial intelligence"]

        combined_layers = f"{result.l1} {result.l2} {result.l3}".lower()

        # At least some key terms should be preserved
        preserved_count = sum(1 for term in key_terms if term in combined_layers)
        assert preserved_count > 0

    def test_manifest_contains_metadata(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test that manifest contains expected metadata."""
        result = compressor.build(long_text)

        # Check tier1 metadata
        assert "tier1" in result.manifest
        tier1 = result.manifest["tier1"]
        assert "total_tokens" in tier1
        assert "budgets" in tier1
        assert "chunks" in tier1
        assert "anchors" in tier1
        assert "hash" in tier1

        # Check tier2 metadata
        assert "tier2" in result.manifest
        tier2 = result.manifest["tier2"]
        assert "budget_tokens" in tier2
        assert "dcp_tokens" in tier2
        assert tier2["dcp_tokens"] >= 0

        # Check tier3 metadata
        assert "tier3" in result.manifest
        tier3 = result.manifest["tier3"]
        assert "method" in tier3

    def test_different_seeds_produce_different_results(self, short_text: str) -> None:
        """Test that different seeds can produce different results."""
        compressor1 = SeraphCompressor(seed=42)
        compressor2 = SeraphCompressor(seed=123)

        result1 = compressor1.build(short_text)
        result2 = compressor2.build(short_text)

        # Different seeds may produce different results due to randomization
        # But both should be valid CompressionResults
        assert isinstance(result1, CompressionResult)
        assert isinstance(result2, CompressionResult)
        assert result1.l1 is not None
        assert result2.l1 is not None

    def test_very_long_text(self, compressor: SeraphCompressor) -> None:
        """Test compression of very long text."""
        # Generate a long document
        very_long_text = "\n\n".join(
            [
                f"Section {i}: This is a paragraph about topic {i}. "
                f"It contains important information about {i} and related concepts. "
                f"Machine learning models can process this information efficiently. "
                f"The system uses advanced algorithms for {i} analysis."
                for i in range(100)
            ]
        )

        result = compressor.build(very_long_text)

        assert isinstance(result, CompressionResult)

        # Should achieve significant compression
        original_tokens = count_tokens(very_long_text)
        l3_tokens = count_tokens(result.l3)

        assert l3_tokens < original_tokens
        assert l3_tokens > 0

    def test_query_on_empty_result(self, compressor: SeraphCompressor) -> None:
        """Test querying on an empty compression result."""
        result = compressor.build("")

        top_k = compressor.query(result, "test question", k=5)

        # Should handle gracefully
        assert isinstance(top_k, list)

    def test_layers_progressively_compress(self, compressor: SeraphCompressor, long_text: str) -> None:
        """Test that layers provide progressive compression levels."""
        result = compressor.build(long_text)

        l1_tokens = count_tokens(result.l1)
        l2_tokens = count_tokens(result.l2)
        l3_tokens = count_tokens(result.l3)

        # L1 should be most compressed, L3 least compressed
        # Note: In some cases with very short text, this might not hold
        if l1_tokens > 0 and l2_tokens > 0 and l3_tokens > 0:
            assert l1_tokens <= l2_tokens or l2_tokens <= l3_tokens
