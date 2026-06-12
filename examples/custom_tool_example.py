"""
Example: Registering a custom tool and running a ReAct agent with it.
"""

from agentic_systems import ReActAgent, ToolRegistry


def weather_mock(city: str) -> str:
    """Return a mock weather report for a city."""
    forecasts = {
        "london": "Overcast, 14°C, light rain expected.",
        "tokyo": "Sunny, 26°C, low humidity.",
        "new york": "Partly cloudy, 21°C.",
    }
    return forecasts.get(city.lower(), f"No forecast available for '{city}'.")


def main() -> None:
    # Build a registry with a custom tool
    registry = ToolRegistry()
    registry.register(weather_mock, description="Get the weather forecast for a city.")

    # Create a ReAct agent
    agent = ReActAgent(
        name="WeatherBot",
        tool_registry=registry,
        verbose=True,
        max_steps=5,
    )

    result = agent.run("What is the weather like in Tokyo?")
    print("\n--- Result ---")
    print(f"Output:  {result.output}")
    print(f"Success: {result.success}")
    print(f"Steps:   {result.steps}")


if __name__ == "__main__":
    main()
