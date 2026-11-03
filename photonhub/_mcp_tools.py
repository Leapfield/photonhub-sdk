"""Registration of local agent tools and resources on a FastMCP server."""

from __future__ import annotations

import json
import subprocess
from functools import wraps

from . import agent_tools, schema
from ._skill import read_skill
from .error_codes import CodedError, error_catalog
from .runners.phsolver import SolverRunError


def coded_errors(fn):
    """Translate tool exceptions into stable, machine-readable failures."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        def message(exc):
            return exc.args[0] if exc.args and isinstance(exc.args[0], str) else str(exc)

        try:
            return fn(*args, **kwargs)
        except CodedError:
            raise
        except SolverRunError as exc:
            id_ = "solver-not-found" if "not found" in str(exc).lower() else "solver-run-failed"
            raise CodedError(id_, message(exc)) from exc
        except RuntimeError as exc:
            id_ = ("solver-run-failed" if fn.__name__ == "run_simulation" and
                   "--capabilities exited" in str(exc) else "internal-error")
            raise CodedError(id_, message(exc)) from exc
        except (json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
            id_ = "solver-run-failed" if fn.__name__ == "run_simulation" else "internal-error"
            raise CodedError(id_, message(exc)) from exc
        except KeyError as exc:
            id_ = "internal-error"
            if fn.__name__ in {"get_spectrum", "field_stats"}:
                data = fn.__globals__.get("_state", {}).get("data")
                monitor = kwargs.get("monitor", args[0] if args else None)
                id_ = ("unknown-monitor" if data is not None and
                       monitor not in data.monitor_names else "invalid-argument")
            raise CodedError(id_, message(exc)) from exc
        except (FileNotFoundError, OSError) as exc:
            id_ = ("result-unreadable" if fn.__name__ in
                   {"open_result", "result_summary", "convergence_report"}
                   else "internal-error")
            raise CodedError(id_, message(exc)) from exc
        except ValueError as exc:
            id_ = ("result-unreadable" if fn.__name__ in
                   {"open_result", "result_summary", "convergence_report"}
                   else "invalid-argument" if fn.__name__ in {"field_stats", "get_spectrum"}
                   else "internal-error")
            raise CodedError(id_, message(exc)) from exc
        except Exception as exc:
            raise CodedError("internal-error", message(exc)) from exc
    return wrapped


def skill_text(name: str = "photonhub") -> str:
    """Read a packaged skill, or return a short fallback for older builds."""
    try:
        return read_skill(name)
    except (FileNotFoundError, OSError):
        return "The PhotonHub agent skill file is not packaged in this build.\n"


def register(mcp, get_current) -> None:
    """Attach six structured tools and five read-only resources."""
    from mcp.types import ToolAnnotations

    annotations = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @mcp.tool(annotations=annotations)
    @coded_errors
    def get_schema(definition: str | None = None) -> agent_tools.SchemaInfo:
        """Return the wire schema or one definition, with its stable digest."""
        return agent_tools.get_schema(definition)

    @mcp.tool(annotations=annotations)
    @coded_errors
    def capabilities() -> agent_tools.Capabilities:
        """Report local solver, device, schema, and safety capabilities."""
        return agent_tools.capabilities()

    @mcp.tool(annotations=annotations)
    @coded_errors
    def validate_spec(spec: str, device: str = "cpu", engine: bool = True) -> agent_tools.ValidationReport:
        """Check a JSON spec in ordered local and engine stages."""
        return agent_tools.validate_spec(spec, device=device, engine=engine)

    @mcp.tool(annotations=annotations)
    @coded_errors
    def estimate(spec: str, cpu_mcells_per_s_low: float = 20.0,
                 cpu_mcells_per_s_high: float = 300.0,
                 device: str = "cpu") -> agent_tools.LocalEstimate:
        """Estimate local size and CPU time from planning throughput bounds."""
        return agent_tools.estimate(spec, cpu_mcells_per_s_low, cpu_mcells_per_s_high,
                                    device=device)

    @mcp.tool(annotations=annotations)
    @coded_errors
    def result_summary(result_dir: str | None = None,
                       data_checks: bool = True) -> agent_tools.ResultSummary:
        """Summarize a result bundle, its provenance, and recorded findings."""
        data = get_current() if result_dir is None else result_dir
        return agent_tools.result_summary(data, data_checks=data_checks)

    @mcp.tool(annotations=annotations)
    @coded_errors
    def convergence_report(result_dir: str | None = None) -> agent_tools.ConvergenceSummary:
        """Read convergence evidence stored with a result bundle."""
        data = get_current() if result_dir is None else result_dir
        return agent_tools.convergence_report(data)

    @mcp.resource("photonhub://schema/simulation_v1.json", mime_type="application/json")
    def schema_resource() -> str:
        """The canonical wire schema text."""
        return schema.schema_text()

    @mcp.resource("photonhub://capabilities", mime_type="application/json")
    def capabilities_resource() -> str:
        """The local capability report."""
        return json.dumps(agent_tools.capabilities().model_dump(mode="json"), indent=2)

    @mcp.resource("photonhub://error-codes", mime_type="application/json")
    def error_codes_resource() -> str:
        """The stable issue-code catalog."""
        return json.dumps(error_catalog(), indent=2)

    @mcp.resource("photonhub://skill/SKILL.md", mime_type="text/markdown")
    def skill_resource() -> str:
        """The packaged agent skill text."""
        return skill_text()

    @mcp.resource("photonhub://skill/import/SKILL.md", mime_type="text/markdown")
    def import_skill_resource() -> str:
        """The packaged skill for rebuilding a simulation from another tool."""
        return skill_text("photonhub-import")
