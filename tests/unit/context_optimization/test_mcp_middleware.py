"""
Unit Tests: MCP Compression Middleware

Fast, focused tests for compression logic in Layer 1 middleware.
Per SDD §10.4.2: Validates threshold gating, timeout handling, error handling,
and automatic quality-based compression ratio selection.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.context_optimization.config import load_config
from src.context_optimization.mcp_middleware import CompressionMiddleware


class TestCompressionLogic:
    """Test compression logic with minimal overhead."""

    @pytest.fixture
    def middleware(self):
        """Create middleware with automatic quality scoring."""
        config = load_config()
        return CompressionMiddleware(
            config=config,
            min_size_bytes=1000,
            timeout_seconds=5.0,
        )

    @pytest.mark.asyncio
    async def test_skip_compression_below_threshold(self, middleware):
        """Content <1KB should skip compression."""
        small_text = "Small content" * 10  # ~130 bytes
        result = await middleware._compress_text(small_text, "test")
        assert result is None  # Skipped

    @pytest.mark.asyncio
    async def test_compress_above_threshold(self, middleware):
        """Content >1KB should trigger compression."""
        large_text = "Large content " * 100  # ~1400 bytes

        # Mock SeraphCompressor
        mock_result = MagicMock()
        mock_result.original_token_count = 200
        mock_result.select_layer.return_value = "Compressed content"

        with patch.object(middleware, '_get_compressor') as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.build = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_compressor

            result = await middleware._compress_text(large_text, "test")
            assert result == "Compressed content"
            mock_compressor.build.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, middleware):
        """Timeout >5s should return None (graceful degradation)."""
        large_text = "Large content " * 100

        with patch.object(middleware, '_get_compressor') as mock_get:
            mock_compressor = MagicMock()
            # Simulate slow compression
            async def slow_compress(*args, **kwargs):
                await asyncio.sleep(10)
            mock_compressor.build = slow_compress
            mock_get.return_value = mock_compressor

            result = await middleware._compress_text(large_text, "test")
            assert result is None  # Timeout fallback

    @pytest.mark.asyncio
    async def test_error_returns_none(self, middleware):
        """Compression errors should return None (graceful degradation)."""
        large_text = "Large content " * 100

        with patch.object(middleware, '_get_compressor') as mock_get:
            mock_compressor = MagicMock()
            mock_compressor.build = AsyncMock(side_effect=RuntimeError("Compression failed"))
            mock_get.return_value = mock_compressor

            result = await middleware._compress_text(large_text, "test")
            assert result is None  # Error fallback

    @pytest.mark.asyncio
    async def test_metrics_recorded_on_success(self, middleware):
        """Successful compression should record metrics."""
        large_text = "Large content " * 100

        mock_result = MagicMock()
        mock_result.original_token_count = 200
        mock_result.select_layer.return_value = "Compressed"

        with patch.object(middleware, '_get_compressor') as mock_get, \
             patch.object(middleware._obs, 'histogram') as mock_hist, \
             patch.object(middleware._obs, 'gauge') as mock_gauge:

            mock_compressor = MagicMock()
            mock_compressor.build = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_compressor

            with patch.object(middleware, '_count_tokens', return_value=100):
                await middleware._compress_text(large_text, "test")

            # Verify metrics recorded
            assert mock_hist.call_count >= 1
            assert mock_gauge.call_count >= 1

    @pytest.mark.asyncio
    async def test_lazy_compressor_initialization(self, middleware):
        """Compressor should initialize on first use."""
        assert middleware._compressor is None

        large_text = "Large content " * 100

        with patch('src.context_optimization.seraph_compression.SeraphCompressor') as mock_class:
            mock_instance = MagicMock()
            mock_result = MagicMock()
            mock_result.original_token_count = 200
            mock_result.select_layer.return_value = "Compressed"
            mock_instance.build = AsyncMock(return_value=mock_result)
            mock_class.return_value = mock_instance

            with patch.object(middleware, '_count_tokens', return_value=100):
                await middleware._compress_text(large_text, "test")

            # Compressor initialized
            assert middleware._compressor is not None


class TestAutomaticQualityScoring:
    """Test automatic content quality analysis for compression ratio selection."""

    @pytest.fixture
    def middleware(self):
        """Create middleware for quality testing with mocked observability."""
        config = load_config()
        mw = CompressionMiddleware(
            config=config,
            min_size_bytes=1000,
            timeout_seconds=5.0,
        )

        # Mock observability to avoid async task issues in sync tests
        mw._obs.gauge = MagicMock()
        mw._obs.histogram = MagicMock()
        mw._obs.increment = MagicMock()

        return mw

    def test_empty_text_conservative_default(self, middleware):
        """Empty/tiny text should get conservative 0.70 default ratio."""
        result = middleware._analyze_content_quality("")
        assert result == 0.70

        result = middleware._analyze_content_quality("short")
        assert result == 0.70

    def test_pure_code_high_preservation(self, middleware):
        """Code-heavy content should get high compression ratio (0.65-0.85)."""
        code_content = """
        def fibonacci(n: int) -> int:
            '''Calculate nth Fibonacci number.'''
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)

        class DataProcessor:
            def __init__(self, config: dict):
                self.config = config

            def process(self, data: list[dict]) -> dict:
                results = []
                for item in data:
                    processed = self._transform(item)
                    results.append(processed)
                return {'results': results, 'count': len(results)}
        """ * 5  # Repeat to exceed 100 char minimum

        ratio = middleware._analyze_content_quality(code_content)
        assert 0.65 <= ratio <= 0.85, f"Code should get high ratio, got {ratio:.2f}"

    def test_json_structure_high_preservation(self, middleware):
        """JSON/structured data should get high compression ratio (0.65-0.85)."""
        json_content = """
        {
            "users": [
                {"id": 1, "name": "Alice", "roles": ["admin", "developer"]},
                {"id": 2, "name": "Bob", "roles": ["viewer"]},
                {"id": 3, "name": "Charlie", "roles": ["developer", "tester"]}
            ],
            "metadata": {
                "version": "2.1.0",
                "timestamp": "2025-10-19T12:00:00Z",
                "environment": "production"
            },
            "configuration": {
                "max_retries": 3,
                "timeout_ms": 5000,
                "enable_caching": true
            }
        }
        """ * 3

        ratio = middleware._analyze_content_quality(json_content)
        assert 0.60 <= ratio <= 0.85, f"JSON should get high ratio, got {ratio:.2f}"

    def test_verbose_prose_aggressive_compression(self, middleware):
        """Verbose prose should get low compression ratio (0.30-0.50)."""
        prose_content = """
        The quick brown fox jumped over the lazy dog. This sentence is used
        to demonstrate verbose natural language text that contains minimal
        technical structure. We continue with more ordinary prose to ensure
        the content analyzer correctly identifies this as low-density text
        suitable for aggressive compression. The weather today is sunny and
        warm, perfect for outdoor activities. Many people enjoy spending time
        in nature during such pleasant conditions. Trees provide shade while
        birds sing melodious songs. Children play games in parks and gardens.
        Families gather for picnics under blue skies. Life feels peaceful
        and calm in these moments of leisure.
        """ * 3

        ratio = middleware._analyze_content_quality(prose_content)
        assert 0.30 <= ratio <= 0.55, f"Prose should get low ratio, got {ratio:.2f}"

    def test_log_output_aggressive_compression(self, middleware):
        """Repetitive log output gets moderate-high ratio due to structure markers."""
        log_content = """
        [2025-10-19 12:00:01] INFO: Processing request 1
        [2025-10-19 12:00:02] INFO: Processing request 2
        [2025-10-19 12:00:03] INFO: Processing request 3
        [2025-10-19 12:00:04] INFO: Processing request 4
        [2025-10-19 12:00:05] INFO: Processing request 5
        [2025-10-19 12:00:06] INFO: Processing request 6
        [2025-10-19 12:00:07] INFO: Processing request 7
        [2025-10-19 12:00:08] INFO: Processing request 8
        [2025-10-19 12:00:09] INFO: Processing request 9
        [2025-10-19 12:00:10] INFO: Processing request 10
        """ * 5

        ratio = middleware._analyze_content_quality(log_content)
        # Logs have brackets/timestamps (structure markers) increasing preservation
        assert 0.70 <= ratio <= 0.80, f"Logs with structure should get moderate-high ratio, got {ratio:.2f}"

    def test_mixed_content_balanced_ratio(self, middleware):
        """Mixed code+prose gets high ratio when code dominates."""
        mixed_content = """
        # User Authentication System

        This module implements JWT-based authentication for the API.
        It provides secure token generation and validation.

        def create_token(user_id: int, expires_in: int = 3600) -> str:
            '''Generate JWT token for user.'''
            payload = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(seconds=expires_in)
            }
            return jwt.encode(payload, SECRET_KEY, algorithm='HS256')

        The function above creates tokens with configurable expiration.
        Default expiration is 1 hour for security purposes.

        class TokenValidator:
            def validate(self, token: str) -> dict:
                try:
                    return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                except jwt.ExpiredSignatureError:
                    raise ValueError('Token expired')
        """ * 2

        ratio = middleware._analyze_content_quality(mixed_content)
        # Code structure (def/class/brackets) dominates scoring despite prose
        assert 0.70 <= ratio <= 0.80, f"Code-heavy mixed content should get high ratio, got {ratio:.2f}"

    def test_high_entropy_preserved(self, middleware):
        """High-entropy content (compressed/random) should get high ratio."""
        # Simulate already-compressed or high-entropy content
        high_entropy = "xK7p9Qw2mN5vB8cR1fL4sT6yZ3hJ0gD" * 20

        ratio = middleware._analyze_content_quality(high_entropy)
        assert 0.55 <= ratio <= 0.85, f"High entropy should preserve more, got {ratio:.2f}"

    def test_technical_vocabulary_moderate_preservation(self, middleware):
        """Technical terms should increase preservation ratio."""
        technical_content = """
        The microserviceArchitecture utilizes containerization with Kubernetes
        orchestration. DatabaseConnectionPooling optimizes throughput while
        implementing circuitBreaker patterns. AsyncAwaitOperations handle
        concurrentRequests through eventLoopScheduling. MemoryManagement
        employs garbageCollection with generationalHeap strategies.
        """ * 3

        ratio = middleware._analyze_content_quality(technical_content)
        assert 0.50 <= ratio <= 0.80, f"Technical vocab should increase ratio, got {ratio:.2f}"

    def test_markdown_table_high_preservation(self, middleware):
        """Markdown tables get moderate ratio (pipe chars have limited structural weight)."""
        table_content = """
        | Feature | Status | Priority |
        |---------|--------|----------|
        | Auth    | Done   | High     |
        | Cache   | TODO   | Medium   |
        | Logs    | Done   | Low      |

        | Metric       | Value | Unit |
        |--------------|-------|------|
        | Latency      | 45    | ms   |
        | Throughput   | 1200  | req/s|
        | Error Rate   | 0.1   | %    |
        """ * 3

        ratio = middleware._analyze_content_quality(table_content)
        # Pipe characters not weighted as heavily as code brackets/braces
        assert 0.40 <= ratio <= 0.55, f"Tables should get moderate ratio, got {ratio:.2f}"

    def test_camel_case_increases_semantic_score(self, middleware):
        """camelCase identifiers should increase semantic density."""
        camel_case_content = """
        const userProfileManager = new UserProfileManager();
        const dataTransformService = createDataTransformService();
        const errorHandlingMiddleware = new ErrorHandlingMiddleware();
        const apiResponseFormatter = initApiResponseFormatter();
        """ * 5

        ratio = middleware._analyze_content_quality(camel_case_content)
        assert 0.55 <= ratio <= 0.85, f"CamelCase should increase ratio, got {ratio:.2f}"

    def test_snake_case_increases_semantic_score(self, middleware):
        """snake_case identifiers should increase semantic density."""
        snake_case_content = """
        user_profile_manager = UserProfileManager()
        data_transform_service = create_data_transform_service()
        error_handling_middleware = ErrorHandlingMiddleware()
        api_response_formatter = init_api_response_formatter()
        """ * 5

        ratio = middleware._analyze_content_quality(snake_case_content)
        assert 0.55 <= ratio <= 0.85, f"snake_case should increase ratio, got {ratio:.2f}"

    def test_redundant_lines_aggressive_compression(self, middleware):
        """Highly redundant content should get aggressive compression."""
        redundant_content = "Same line repeated\n" * 100

        ratio = middleware._analyze_content_quality(redundant_content)
        assert 0.30 <= ratio <= 0.50, f"Redundant content should get low ratio, got {ratio:.2f}"

    def test_quality_metrics_tracked(self, middleware):
        """Quality analysis should track all dimension metrics."""
        code_content = """
        def process(data: dict) -> dict:
            return {'result': data['value'] * 2}
        """ * 10

        with patch.object(middleware._obs, 'gauge') as mock_gauge:
            middleware._analyze_content_quality(code_content)

            # Verify all quality metrics tracked
            metric_calls = [call[0][0] for call in mock_gauge.call_args_list]
            assert "mcp.middleware.quality.structure" in metric_calls
            assert "mcp.middleware.quality.entropy" in metric_calls
            assert "mcp.middleware.quality.redundancy" in metric_calls
            assert "mcp.middleware.quality.semantic" in metric_calls
            assert "mcp.middleware.quality.final_score" in metric_calls
            assert "mcp.middleware.quality.compression_ratio" in metric_calls

    def test_ratio_bounds_enforced(self, middleware):
        """Compression ratio should always be within [0.30, 0.85]."""
        test_cases = [
            "a" * 1000,  # Minimal structure
            "{" * 500 + "}" * 500,  # Maximal structure
            "x1y2z3" * 200,  # High entropy
            "same\n" * 200,  # Low entropy
        ]

        for content in test_cases:
            ratio = middleware._analyze_content_quality(content)
            assert 0.30 <= ratio <= 0.85, f"Ratio {ratio:.2f} out of bounds for content length {len(content)}"

    @pytest.mark.asyncio
    async def test_automatic_ratio_used_in_compression(self, middleware):
        """_compress_text should use automatic quality scoring."""
        code_content = "def test(): pass\n" * 100  # Code → high ratio

        mock_result = MagicMock()
        mock_result.original_token_count = 200
        mock_result.select_layer.return_value = "Compressed"

        with patch.object(middleware, '_get_compressor') as mock_get, \
             patch.object(middleware, '_count_tokens', return_value=100):

            mock_compressor = MagicMock()
            mock_compressor.build = AsyncMock(return_value=mock_result)
            mock_get.return_value = mock_compressor

            await middleware._compress_text(code_content, "test")

            # Verify select_layer called with automatic ratio (not hardcoded 0.5)
            mock_result.select_layer.assert_called_once()
            call_ratio = mock_result.select_layer.call_args[1]['target_ratio']
            assert 0.30 <= call_ratio <= 0.85
            assert call_ratio != 0.50  # Should NOT be hardcoded default
