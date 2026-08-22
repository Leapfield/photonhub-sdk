"""Configuration for the cloud client (``ph.web``).

``configure(api_key=..., url=...)`` sets the active config; values fall back to
``$PHOTONHUB_API_KEY`` / ``$PHOTONHUB_URL``, mirroring ``find_solver``'s
explicit→environment precedence. A missing required value is an error, not a
silent default. ``WebError`` is raised for config/transport/auth/result-transfer
problems — distinct from ``SolverRunError``, which is reserved for a simulation
actually failing, so the two are never confused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import ipaddress
import math
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from .._env import env
from ..bundle import (
    DEFAULT_MAX_COMPRESSED_BYTES,
    DEFAULT_MAX_EXPANDED_BYTES,
    DEFAULT_MAX_MEMBERS,
)

#: Conventional local dev-server endpoint. Never applied implicitly:
#: ``configure()`` requires an explicit ``url=`` / ``$PHOTONHUB_URL``, so a
#: half-configured client (key but no URL) fails with an actionable message
#: instead of dialing localhost and dying with a raw connection error.
DEFAULT_URL = "http://localhost:8000"


class WebError(RuntimeError):
    """A cloud client/transport/auth/result-transfer error — NOT a simulation
    failure (that is ``SolverRunError``). Accepted jobs expose ``job_id`` for
    safe resume without a duplicate paid submission."""

    def __init__(self, message: str, *, status_code: Optional[int] = None,
                 body=None, job_id: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        # Once a submission is accepted, transport/result failures remain
        # recoverable: callers can resume this exact server-side job instead
        # of risking a duplicate paid submission.
        self.job_id = job_id


@dataclass
class WebConfig:
    url: str
    #: Kept out of the generated repr: a bare ``get_config()`` in a notebook
    #: cell would otherwise write the live key into committed output.
    api_key: str = field(repr=False)
    cache_dir: Path
    poll_interval_s: float = 2.0
    poll_backoff_max_s: float = 15.0
    request_timeout_s: float = 30.0
    allow_insecure_http: bool = False
    max_bundle_download_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES
    max_bundle_extract_bytes: int = DEFAULT_MAX_EXPANDED_BYTES
    max_bundle_members: int = DEFAULT_MAX_MEMBERS

    def __post_init__(self) -> None:
        if (not isinstance(self.api_key, str) or not self.api_key
                or any(ch.isspace() for ch in self.api_key)):
            raise WebError("api_key must be a non-empty token without whitespace")
        self.url = _validate_url(self.url, self.allow_insecure_http)
        self.cache_dir = Path(self.cache_dir)
        self.poll_interval_s = _finite_seconds(
            self.poll_interval_s, "poll_interval_s", allow_zero=True)
        self.poll_backoff_max_s = _finite_seconds(
            self.poll_backoff_max_s, "poll_backoff_max_s", allow_zero=True)
        self.request_timeout_s = _finite_seconds(
            self.request_timeout_s, "request_timeout_s", allow_zero=False)
        if self.poll_backoff_max_s < self.poll_interval_s:
            raise ValueError(
                "poll_backoff_max_s must be >= poll_interval_s")
        self.max_bundle_download_bytes = _positive_int(
            self.max_bundle_download_bytes, "max_bundle_download_bytes")
        self.max_bundle_extract_bytes = _positive_int(
            self.max_bundle_extract_bytes, "max_bundle_extract_bytes")
        self.max_bundle_members = _positive_int(
            self.max_bundle_members, "max_bundle_members")


_CONFIG: Optional[WebConfig] = None


def _finite_seconds(value, label: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite " +
                         ("non-negative" if allow_zero else "positive") +
                         " number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite " +
                         ("non-negative" if allow_zero else "positive") +
                         " number") from exc
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{label} must be a finite " +
                         ("non-negative" if allow_zero else "positive") +
                         " number")
    return number


def _validate_url(url, allow_insecure_http) -> str:
    """Check a service URL and return it with any trailing slash removed."""
    if not isinstance(url, str) or not url:
        raise ValueError("url must be a non-empty http(s) URL")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        # Accessing .port performs urllib's numeric/range validation.
        parsed.port
    except ValueError as exc:
        raise ValueError("url has an invalid host or port") from exc
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or hostname is None or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or any(ch.isspace() for ch in url)):
        raise ValueError(
            "url must be an http(s) service URL without credentials, "
            "query, fragment, or whitespace")
    if not isinstance(allow_insecure_http, bool):
        raise ValueError("allow_insecure_http must be a boolean")
    if (parsed.scheme == "http" and not allow_insecure_http
            and not _is_loopback_host(hostname)):
        raise ValueError(
            "non-loopback cloud URLs require HTTPS; pass "
            "allow_insecure_http=True only for an explicitly trusted "
            "development network")
    return url.rstrip("/")


def _is_loopback_host(hostname: str) -> bool:
    host = hostname.lower()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _default_cache_dir() -> Path:
    val = env("CACHE_DIR")
    if val:
        return Path(val)
    return Path.home() / ".cache" / "photonhub" / "jobs"


def configure(api_key: Optional[str] = None, url: Optional[str] = None, *,
              cache_dir=None, poll_interval_s: float = 2.0,
              poll_backoff_max_s: float = 15.0,
              request_timeout_s: float = 30.0,
              allow_insecure_http: bool = False,
              max_bundle_download_bytes: int = DEFAULT_MAX_COMPRESSED_BYTES,
              max_bundle_extract_bytes: int = DEFAULT_MAX_EXPANDED_BYTES,
              max_bundle_members: int = DEFAULT_MAX_MEMBERS) -> WebConfig:
    """Set the active cloud configuration. Returns it for inspection."""
    global _CONFIG
    key = env("API_KEY") if api_key is None else api_key
    if not key:
        raise WebError(
            "no API key: pass api_key= or set $PHOTONHUB_API_KEY "
            "(create one with ph.web.create_api_key after signing in)")
    base = url
    if base is None:
        # An explicitly passed (even invalid) url= falls through to
        # WebConfig's ValueError; only a genuinely *absent* URL raises here.
        base = env("URL")
        if not base:
            raise WebError(
                "no service URL: pass url= or set $PHOTONHUB_URL. During the "
                "beta the endpoint is issued by the operator together with "
                "your API key (see docs/cloud.md). For a local development "
                f"server, pass url={DEFAULT_URL!r} explicitly.")
        try:
            _validate_url(base, allow_insecure_http)
        except ValueError as exc:
            raise WebError(
                f"$PHOTONHUB_URL is not a usable service URL ({base!r}): {exc}. "
                "Expected a bare https:// origin, e.g. "
                "'https://api.example.com'.") from exc
    cache = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    _CONFIG = WebConfig(
        url=base, api_key=key, cache_dir=cache,
        poll_interval_s=poll_interval_s, poll_backoff_max_s=poll_backoff_max_s,
        request_timeout_s=request_timeout_s,
        allow_insecure_http=allow_insecure_http,
        max_bundle_download_bytes=max_bundle_download_bytes,
        max_bundle_extract_bytes=max_bundle_extract_bytes,
        max_bundle_members=max_bundle_members,
    )
    return _CONFIG


def get_config() -> WebConfig:
    """The active config, building one from the environment on first use."""
    if _CONFIG is not None:
        return _CONFIG
    if env("API_KEY"):
        return configure()
    raise WebError(
        "photonhub.web is not configured; call "
        "ph.web.configure(api_key=..., url=...) or set "
        "$PHOTONHUB_API_KEY + $PHOTONHUB_URL")


def reset() -> None:
    """Clear the active config (mainly for tests)."""
    global _CONFIG
    _CONFIG = None
