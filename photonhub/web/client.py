"""Minimal HTTP client over ``urllib`` (stdlib transport; certifi only as a
CA-bundle fallback for root-less interpreters).

Bearer-authenticated JSON requests with capped-backoff retry on 5xx/network
errors (idempotent GETs). 4xx errors raise immediately as ``WebError`` carrying
the parsed ``detail``. ``urllib`` transparently follows the 302 that the result
endpoint returns in prod (to a signed object-store URL).
"""

from __future__ import annotations

import json
import io
import http.client
from pathlib import Path
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from ._ids import validate_job_id
from .config import WebConfig, WebError

_RETRIES = 3
_DOWNLOAD_CHUNK = 1024 * 1024
_MAX_ERROR_BODY_BYTES = 1024 * 1024
_MAX_JSON_BODY_BYTES = 16 * 1024 * 1024


class _NoHttpsDowngrade(urllib.request.HTTPRedirectHandler):
    """Allow signed-result redirects without permitting HTTPS downgrade."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        source_scheme = urllib.parse.urlsplit(req.full_url).scheme.lower()
        target_scheme = urllib.parse.urlsplit(newurl).scheme.lower()
        if target_scheme not in ("http", "https") or (
                source_scheme == "https" and target_scheme != "https"):
            raise urllib.error.HTTPError(
                newurl, code, "unsafe redirect target", headers, fp)
        return super().redirect_request(
            req, fp, code, msg, headers, newurl)


def _https_context() -> ssl.SSLContext:
    """Default-verification TLS context, falling back to certifi's CA bundle
    only when the interpreter's default context trusts no CAs at all (e.g. a
    python.org macOS build whose "Install Certificates" step never ran, where
    every HTTPS request would fail with CERTIFICATE_VERIFY_FAILED).
    Verification is never relaxed: without certifi a CA-less context still
    fails loudly."""
    ctx = ssl.create_default_context()
    if ctx.cert_store_stats().get("x509_ca", 0) == 0:
        try:
            import certifi
        except ImportError:
            return ctx
        ctx.load_verify_locations(cafile=certifi.where())
    return ctx


_HTTPS_OPENER = urllib.request.build_opener(
    urllib.request.HTTPSHandler(context=_https_context()), _NoHttpsDowngrade())


def _parse_detail(body: str) -> Any:
    try:
        obj = json.loads(body)
        return obj.get("detail", obj) if isinstance(obj, dict) else obj
    except Exception:
        return body


class HttpClient:
    def __init__(self, cfg: WebConfig):
        self.cfg = cfg

    def _open(self, method: str, path: str, *, body: Optional[dict] = None,
              deadline: Optional[float] = None):
        url = self.cfg.url + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        # The result endpoint redirects to a signed object-store URL. A normal
        # Request header is copied by urllib's redirect handler, which would
        # disclose the API bearer token cross-origin. Unredirected headers are
        # sent to the API origin but deliberately omitted from redirected
        # requests.
        req.add_unredirected_header(
            "Authorization", f"Bearer {self.cfg.api_key}")

        last: Optional[Exception] = None
        max_attempts = _RETRIES if method == "GET" else 1
        for attempt in range(max_attempts):
            request_timeout = self.cfg.request_timeout_s
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("HTTP request deadline expired")
                request_timeout = min(request_timeout, remaining)
            try:
                if urllib.parse.urlsplit(url).scheme == "https":
                    return _HTTPS_OPENER.open(req, timeout=request_timeout)
                return urllib.request.urlopen(req, timeout=request_timeout)
            except urllib.error.HTTPError as e:
                try:
                    try:
                        raw_error = e.read(_MAX_ERROR_BODY_BYTES + 1)
                        if len(raw_error) > _MAX_ERROR_BODY_BYTES:
                            detail = (
                                "response error body exceeded the 1 MiB limit")
                        else:
                            detail = _parse_detail(
                                raw_error.decode(errors="replace"))
                    except (OSError, http.client.HTTPException):
                        detail = "response error body could not be read"
                finally:
                    e.close()
                if 500 <= e.code < 600 and attempt < max_attempts - 1:
                    last = e
                    self._retry_sleep(0.5 * (attempt + 1), deadline)
                    continue
                raise WebError(f"{method} {path} -> HTTP {e.code}",
                               status_code=e.code, body=detail)
            except urllib.error.URLError as e:
                last = e
                if attempt < max_attempts - 1:
                    self._retry_sleep(0.5 * (attempt + 1), deadline)
                    continue
                raise WebError(f"{method} {path} failed: {e.reason}")
        raise WebError(f"{method} {path} failed after retries: {last}")

    @staticmethod
    def _retry_sleep(delay: float, deadline: Optional[float]) -> None:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HTTP request deadline expired")
            delay = min(delay, remaining)
        time.sleep(delay)

    @staticmethod
    def _job_path(job_id: object) -> str:
        safe = validate_job_id(job_id)
        return urllib.parse.quote(safe, safe="")

    def get_json(self, path: str, *, deadline: Optional[float] = None) -> dict:
        try:
            with self._open("GET", path, deadline=deadline) as response:
                raw = self._read_limited(
                    response, path, _MAX_JSON_BODY_BYTES, method="GET")
        except WebError:
            raise
        except TimeoutError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise WebError(f"GET {path} failed while reading: {exc}") from exc
        return self._decode_json(raw, f"GET {path}")

    def post_json(self, path: str, body: dict) -> dict:
        try:
            with self._open("POST", path, body=body) as response:
                raw = self._read_limited(
                    response, path, _MAX_JSON_BODY_BYTES, method="POST")
        except WebError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise WebError(f"POST {path} failed while reading: {exc}") from exc
        return self._decode_json(raw, f"POST {path}")

    def delete(self, path: str) -> int:
        with self._open("DELETE", path) as r:
            return r.status

    @staticmethod
    def _content_length(response, path: str) -> Optional[int]:
        headers = getattr(response, "headers", None)
        raw = headers.get("Content-Length") if headers is not None else None
        if raw is None:
            return None
        try:
            size = int(raw)
        except (TypeError, ValueError) as exc:
            raise WebError(
                f"GET {path} returned an invalid Content-Length") from exc
        if size < 0:
            raise WebError(f"GET {path} returned an invalid Content-Length")
        return size

    @staticmethod
    def _download_limit(value: Optional[int], default: int) -> int:
        limit = default if value is None else value
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("max_bytes must be a positive integer")
        return limit

    def _read_limited(self, response, path: str, limit: int, *,
                      method: str = "GET") -> bytes:
        declared = self._content_length(response, path)
        if declared is not None and declared > limit:
            raise WebError(
                f"{method} {path} exceeds the {limit}-byte response limit")
        output = io.BytesIO()
        total = 0
        while True:
            chunk = response.read(min(_DOWNLOAD_CHUNK, limit - total + 1))
            if not chunk:
                if declared is not None and total != declared:
                    raise WebError(
                        f"{method} {path} ended after {total} bytes; "
                        f"Content-Length declared {declared}")
                return output.getvalue()
            total += len(chunk)
            if total > limit:
                raise WebError(
                    f"{method} {path} exceeds the {limit}-byte response limit")
            output.write(chunk)

    @staticmethod
    def _decode_json(raw: bytes, operation: str) -> dict:
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, ValueError, RecursionError) as exc:
            raise WebError(f"{operation} returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise WebError(f"{operation} returned a non-object JSON response")
        return value

    def get_bytes(self, path: str, *, max_bytes: Optional[int] = None) -> bytes:
        """Read a response with a hard byte ceiling.

        This compatibility helper remains useful for small endpoints and test
        transports. Large result bundles use :meth:`get_to_file` so they are
        never duplicated in memory.
        """
        limit = self._download_limit(
            max_bytes, self.cfg.max_bundle_download_bytes)
        try:
            with self._open("GET", path) as response:
                return self._read_limited(response, path, limit)
        except WebError:
            raise
        except (OSError, http.client.HTTPException) as exc:
            raise WebError(f"GET {path} failed while reading: {exc}") from exc

    def get_to_file(self, path: str, dest, *,
                    max_bytes: Optional[int] = None) -> int:
        """Stream one GET response into ``dest`` with a hard byte ceiling.

        A rejected, truncated, or failed download leaves no partial file.
        Returns the number of bytes written.
        """
        limit = self._download_limit(
            max_bytes, self.cfg.max_bundle_download_bytes)
        dest = Path(dest)
        try:
            with self._open("GET", path) as response:
                declared = self._content_length(response, path)
                if declared is not None and declared > limit:
                    raise WebError(
                        f"GET {path} exceeds the {limit}-byte download limit")
                total = 0
                with dest.open("wb") as output:
                    while True:
                        chunk = response.read(
                            min(_DOWNLOAD_CHUNK, limit - total + 1))
                        if not chunk:
                            if declared is not None and total != declared:
                                raise WebError(
                                    f"GET {path} ended after {total} bytes; "
                                    f"Content-Length declared {declared}")
                            return total
                        total += len(chunk)
                        if total > limit:
                            raise WebError(
                                f"GET {path} exceeds the {limit}-byte "
                                "download limit")
                        output.write(chunk)
        except (OSError, http.client.HTTPException) as exc:
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            raise WebError(f"GET {path} failed while reading: {exc}") from exc
        except BaseException:
            try:
                dest.unlink()
            except FileNotFoundError:
                pass
            raise

    # --- typed endpoint helpers -------------------------------------------

    def whoami(self) -> dict:
        return self.get_json("/v1/auth/whoami")

    def account(self) -> dict:
        return self.get_json("/v1/account")

    def estimate(self, spec: dict, *, device=None, solver=None) -> dict:
        body = {"spec": spec}
        if device is not None:
            body["device"] = device
        if solver is not None:
            body["solver"] = solver
        return self.post_json("/v1/estimate", body)

    def create_api_key(self, name: str = "default") -> dict:
        return self.post_json("/v1/keys", {"name": name})

    def list_gpus(self) -> list:
        return self.get_json("/v1/gpus").get("gpus", [])

    def list_jobs(self) -> list[dict]:
        payload = self.get_json("/v1/jobs")
        jobs = payload.get("jobs")
        if (not isinstance(jobs, list)
                or any(not isinstance(record, dict) for record in jobs)):
            raise WebError("GET /v1/jobs returned an invalid 'jobs' list")
        return jobs

    def submit_job(self, spec: dict, *, name=None, device=None,
                   solver=None, quote_id=None) -> dict:
        body: dict = {"spec": spec}
        if name is not None:
            body["name"] = name
        if device is not None:
            body["device"] = device
        if solver is not None:
            body["solver"] = solver
        if quote_id is not None:
            body["quote_id"] = quote_id
        return self.post_json("/v1/jobs", body)

    def get_job(self, job_id: str, *, deadline: Optional[float] = None) -> dict:
        return self.get_json(
            f"/v1/jobs/{self._job_path(job_id)}", deadline=deadline)

    def cancel_job(self, job_id: str) -> dict:
        return self.post_json(f"/v1/jobs/{self._job_path(job_id)}/cancel", {})

    def download_result(self, job_id: str) -> bytes:
        return self.get_bytes(f"/v1/jobs/{self._job_path(job_id)}/result")

    def download_result_to(self, job_id: str, dest, *,
                           max_bytes: Optional[int] = None) -> int:
        return self.get_to_file(
            f"/v1/jobs/{self._job_path(job_id)}/result", dest,
            max_bytes=max_bytes)
