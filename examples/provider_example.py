"""
Provider Usage Example

Demonstrates how to use the AI model provider system in Seraph MCP.

This example shows:
- Creating and configuring providers
- Making completion requests
- Handling multiple providers
- Cost estimation
- Health checks
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from providers import (
    CompletionRequest,
    ProviderConfig,
    create_provider,
    list_providers,
    close_all_providers,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def example_basic_completion():
    """Example: Basic completion with OpenAI."""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Completion with OpenAI")
    logger.info("=" * 60)

    # Create OpenAI provider
    config = ProviderConfig(
        api_key="sk-your-api-key-here",  # Replace with real key
        timeout=30.0,
        max_retries=3,
    )

    try:
        provider = create_provider("openai", config)
        logger.info(f"Created provider: {provider}")

        # Create completion request
        request = CompletionRequest(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say hello in 5 words or less."}
            ],
            temperature=0.7,
            max_tokens=20,
        )

        # Generate completion
        response = await provider.complete(request)

        logger.info(f"Response: {response.content}")
        logger.info(f"Usage: {response.usage}")
        logger.info(f"Cost: ${response.cost_usd:.6f}")
        logger.info(f"Latency: {response.latency_ms:.2f}ms")

    except Exception as e:
        logger.error(f"Error: {e}")


async def example_multiple_providers():
    """Example: Working with multiple providers."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Multiple Providers")
    logger.info("=" * 60)

    # Create multiple providers
    providers = {}

    # OpenAI
    try:
        providers["openai"] = create_provider(
            "openai",
            ProviderConfig(api_key="sk-your-openai-key")
        )
        logger.info("✓ OpenAI provider created")
    except Exception as e:
        logger.warning(f"✗ OpenAI provider failed: {e}")

    # Anthropic
    try:
        providers["anthropic"] = create_provider(
            "anthropic",
            ProviderConfig(api_key="sk-ant-your-anthropic-key")
        )
        logger.info("✓ Anthropic provider created")
    except Exception as e:
        logger.warning(f"✗ Anthropic provider failed: {e}")

    # Gemini
    try:
        providers["gemini"] = create_provider(
            "gemini",
            ProviderConfig(api_key="AIza-your-gemini-key")
        )
        logger.info("✓ Gemini provider created")
    except Exception as e:
        logger.warning(f"✗ Gemini provider failed: {e}")

    # List all active providers
    active = list_providers()
    logger.info(f"\nActive providers: {list(active.keys())}")


async def example_cost_estimation():
    """Example: Cost estimation before making requests."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Cost Estimation")
    logger.info("=" * 60)

    config = ProviderConfig(api_key="sk-your-api-key-here")

    try:
        provider = create_provider("openai", config)

        # Estimate costs for different models
        models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        input_tokens = 1000
        output_tokens = 500

        logger.info(f"\nEstimating costs for {input_tokens} input + {output_tokens} output tokens:")
        logger.info("-" * 60)

        for model in models:
            cost = await provider.estimate_cost(model, input_tokens, output_tokens)
            logger.info(f"{model:20s} → ${cost:.6f}")

    except Exception as e:
        logger.error(f"Error: {e}")


async def example_model_info():
    """Example: Getting model information."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Model Information")
    logger.info("=" * 60)

    config = ProviderConfig(api_key="sk-your-api-key-here")

    try:
        provider = create_provider("openai", config)

        # List all available models
        models = await provider.list_models()
        logger.info(f"\nAvailable models: {len(models)}")
        logger.info("-" * 60)

        for model in models:
            logger.info(f"\n{model.display_name} ({model.model_id})")
            logger.info(f"  Context window: {model.context_window:,} tokens")
            logger.info(f"  Max output: {model.max_output_tokens:,} tokens")
            logger.info(f"  Cost: ${model.input_cost_per_1k:.4f} / ${model.output_cost_per_1k:.4f} per 1K")
            logger.info(f"  Capabilities: {', '.join(model.capabilities)}")

    except Exception as e:
        logger.error(f"Error: {e}")


async def example_health_check():
    """Example: Provider health checks."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 5: Health Checks")
    logger.info("=" * 60)

    providers_to_check = [
        ("openai", ProviderConfig(api_key="sk-your-openai-key")),
        ("anthropic", ProviderConfig(api_key="sk-ant-your-anthropic-key")),
        ("gemini", ProviderConfig(api_key="AIza-your-gemini-key")),
    ]

    for name, config in providers_to_check:
        try:
            provider = create_provider(name, config)
            is_healthy = await provider.health_check()
            status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
            logger.info(f"{name:15s} → {status}")
        except Exception as e:
            logger.warning(f"{name:15s} → ✗ Failed: {e}")


async def example_openai_compatible():
    """Example: Using OpenAI-compatible provider (LM Studio, Ollama, etc.)."""
    logger.info("\n" + "=" * 60)
    logger.info("Example 6: OpenAI-Compatible Provider (Local Models)")
    logger.info("=" * 60)

    # Configure for local LM Studio instance
    config = ProviderConfig(
        api_key="not-needed",  # Local services often don't need API keys
        base_url="http://localhost:1234/v1",  # LM Studio default
        timeout=60.0,  # Local inference might take longer
    )

    try:
        provider = create_provider("openai-compatible", config)
        logger.info(f"Created provider: {provider}")

        # List available local models
        models = await provider.list_models()
        logger.info(f"\nLocal models available: {len(models)}")
        for model in models:
            logger.info(f"  - {model.model_id}")

        # Make a completion request
        if models:
            request = CompletionRequest(
                model=models[0].model_id,
                messages=[
                    {"role": "user", "content": "Hello! Introduce yourself."}
                ],
                temperature=0.7,
                max_tokens=100,
            )

            response = await provider.complete(request)
            logger.info(f"\nResponse: {response.content}")
            logger.info(f"Latency: {response.latency_ms:.2f}ms")
            logger.info(f"Cost: ${response.cost_usd:.6f} (local is free!)")

    except Exception as e:
        logger.error(f"Error: {e}")
        logger.info("\nNote: Make sure LM Studio (or similar) is running at http://localhost:1234")


async def main():
    """Run all examples."""
    logger.info("Seraph MCP Provider System Examples")
    logger.info("=" * 60)
    logger.info("\nNote: Replace API keys with real keys to run examples")
    logger.info("      Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY in .env\n")

    try:
        # Run examples (comment out as needed)
        await example_basic_completion()
        await example_multiple_providers()
        await example_cost_estimation()
        await example_model_info()
        await example_health_check()
        await example_openai_compatible()

    finally:
        # Clean up all providers
        logger.info("\n" + "=" * 60)
        logger.info("Cleaning up providers...")
        await close_all_providers()
        logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
