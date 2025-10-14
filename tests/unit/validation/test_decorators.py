"""
Tests for Validation Decorators

Comprehensive test coverage for @validate_input decorator.

P0 Phase 1b Testing:
- Valid input handling
- Invalid input detection
- Structured error responses
- Observability integration
- Async and sync function support
"""

import pytest
from pydantic import BaseModel, Field

from src.errors import ErrorCode
from src.validation.decorators import validate_input


class SampleInput(BaseModel):
    """Sample validation schema for testing."""

    name: str = Field(..., min_length=1, max_length=100)
    age: int = Field(..., ge=0, le=150)
    email: str | None = Field(default=None)


class TestValidateInputDecorator:
    """Test @validate_input decorator."""

    @pytest.mark.asyncio
    async def test_valid_async_input(self):
        """Test decorator with valid async input."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age, "email": email}

        result = await sample_func(name="Alice", age=30, email="alice@example.com")

        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["email"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_invalid_async_input_missing_field(self):
        """Test decorator with missing required field."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        result = await sample_func(name="Alice")

        # Should return error response
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "validation_errors" in result["details"]
        assert len(result["details"]["validation_errors"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_async_input_type_error(self):
        """Test decorator with wrong type."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        result = await sample_func(name="Alice", age="not_an_int")

        # Should return error response
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "validation_errors" in result["details"]

    @pytest.mark.asyncio
    async def test_invalid_async_input_constraint_violation(self):
        """Test decorator with constraint violation."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        # Age exceeds max (150)
        result = await sample_func(name="Alice", age=200)

        # Should return error response
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "validation_errors" in result["details"]
        errors = result["details"]["validation_errors"]
        assert any("age" in err["field"] for err in errors)

    @pytest.mark.asyncio
    async def test_invalid_async_input_string_length(self):
        """Test decorator with string length violation."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        # Name too long
        result = await sample_func(name="A" * 200, age=30)

        # Should return error response
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "validation_errors" in result["details"]
        errors = result["details"]["validation_errors"]
        assert any("name" in err["field"] for err in errors)

    @pytest.mark.asyncio
    async def test_valid_async_optional_field_omitted(self):
        """Test decorator with optional field omitted."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age, "email": email}

        # Email is optional
        result = await sample_func(name="Alice", age=30)

        assert result["name"] == "Alice"
        assert result["age"] == 30
        assert result["email"] is None

    @pytest.mark.asyncio
    async def test_error_response_structure(self):
        """Test error response has correct structure."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        result = await sample_func(age="invalid")

        # Verify error response structure
        assert "success" in result
        assert result["success"] is False
        assert "error_code" in result
        assert result["error_code"] == ErrorCode.INVALID_INPUT
        assert "message" in result
        assert "details" in result
        assert "validation_errors" in result["details"]
        assert "function" in result["details"]

        # Verify validation error structure
        errors = result["details"]["validation_errors"]
        assert len(errors) > 0
        for error in errors:
            assert "field" in error
            assert "message" in error
            assert "type" in error

    def test_valid_sync_input(self):
        """Test decorator with valid sync input."""

        @validate_input(SampleInput)
        def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age, "email": email}

        result = sample_func(name="Bob", age=25)

        assert result["name"] == "Bob"
        assert result["age"] == 25

    def test_invalid_sync_input(self):
        """Test decorator with invalid sync input."""

        @validate_input(SampleInput)
        def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        result = sample_func(name="", age=25)

        # Empty name violates min_length constraint
        assert result["success"] is False
        assert result["error_code"] == ErrorCode.INVALID_INPUT

    @pytest.mark.asyncio
    async def test_multiple_validation_errors(self):
        """Test decorator with multiple validation errors."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        # Both name and age are invalid
        result = await sample_func(name="", age=-5)

        assert result["success"] is False
        errors = result["details"]["validation_errors"]
        assert len(errors) >= 2  # At least 2 errors (name and age)

    @pytest.mark.asyncio
    async def test_function_name_in_error(self):
        """Test function name is included in error details."""

        @validate_input(SampleInput)
        async def my_special_function(name: str, age: int, email: str | None = None):
            return {"name": name, "age": age}

        result = await my_special_function(name="", age=30)

        assert result["details"]["function"] == "my_special_function"

    @pytest.mark.asyncio
    async def test_decorator_preserves_function_metadata(self):
        """Test decorator preserves original function metadata."""

        @validate_input(SampleInput)
        async def documented_function(name: str, age: int, email: str | None = None):
            """This function has documentation."""
            return {"name": name, "age": age}

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."

    @pytest.mark.asyncio
    async def test_validation_with_extra_kwargs(self):
        """Test validation handles extra kwargs gracefully."""

        @validate_input(SampleInput)
        async def sample_func(name: str, age: int, email: str | None = None, **kwargs):
            return {"name": name, "age": age, "extra": kwargs}

        # Extra kwargs not in schema should be handled
        result = await sample_func(name="Alice", age=30, extra_field="ignored")

        # Should succeed with valid required fields
        assert result["name"] == "Alice"
        assert result["age"] == 30


class EmptyInput(BaseModel):
    """Schema with no required fields."""

    pass


class TestValidateInputEdgeCases:
    """Test edge cases for validation decorator."""

    @pytest.mark.asyncio
    async def test_empty_schema(self):
        """Test decorator with empty schema."""

        @validate_input(EmptyInput)
        async def sample_func():
            return {"success": True}

        result = await sample_func()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_validation_with_defaults(self):
        """Test validation uses default values from schema."""

        class InputWithDefaults(BaseModel):
            name: str
            count: int = 10

        @validate_input(InputWithDefaults)
        async def sample_func(name: str, count: int = 10):
            return {"name": name, "count": count}

        result = await sample_func(name="Test")
        assert result["name"] == "Test"
        assert result["count"] == 10

    @pytest.mark.asyncio
    async def test_nested_field_path_in_error(self):
        """Test nested field paths are properly formatted in errors."""

        class NestedModel(BaseModel):
            inner: str

        class OuterModel(BaseModel):
            nested: NestedModel

        @validate_input(OuterModel)
        async def sample_func(nested: dict):
            return nested

        result = await sample_func(nested={"inner": 123})  # Wrong type

        # Should have field path like "nested -> inner"
        errors = result["details"]["validation_errors"]
        assert len(errors) > 0
        # Field path should indicate nesting
        assert any("nested" in err["field"] for err in errors)
