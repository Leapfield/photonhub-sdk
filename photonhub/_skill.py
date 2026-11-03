"""Read the agent skills bundled with the local SDK."""

from importlib import resources


def read_skill(name: str = "photonhub") -> str:
    """Return a packaged skill's text without loading optional MCP support."""
    return resources.files("photonhub").joinpath(
        f"skills/{name}/SKILL.md").read_text(encoding="utf-8")
