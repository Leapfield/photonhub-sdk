"""Read the agent skill bundled with the local SDK."""

from importlib import resources


def read_skill() -> str:
    """Return the packaged skill text without loading optional MCP support."""
    return resources.files("photonhub").joinpath(
        "skills/photonhub/SKILL.md").read_text(encoding="utf-8")
