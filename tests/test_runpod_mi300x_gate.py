from __future__ import annotations

import copy
import datetime as dt
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "runpod_mi300x_gate.py"
WORKFLOW = ROOT / ".github" / "workflows" / "mi300x-hardware-gate.yml"

spec = importlib.util.spec_from_file_location("runpod_mi300x_gate", SCRIPT)
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)

IMAGE = "ghcr.io/leapfield/photonhub-solver@sha256:" + "a" * 64
SOURCE_SHA = "b" * 40
PUT_URL = "https://objects.example.test/gate.json?put=secret-capability"
GET_URL = "https://objects.example.test/gate.json?get=secret-capability"
ATTEMPT_ID = "test-attempt-1"
TERMINATE_AFTER = dt.datetime(
    2026, 7, 22, 10, 30, tzinfo=dt.timezone.utc
)


def _args(*extra: str):
    return gate.build_parser().parse_args([IMAGE, SOURCE_SHA, *extra])


def _config(
    *extra: str,
    api_key: str | None = "provider-secret",
    put_url: str = PUT_URL,
    get_url: str = GET_URL,
    registry_auth_id: str | None = None,
):
    environ = {
        "PHOTONHUB_EVIDENCE_PUT_URL": put_url,
        "PHOTONHUB_EVIDENCE_GET_URL": get_url,
        "PHOTONHUB_GATE_ATTEMPT_ID": ATTEMPT_ID,
    }
    if api_key is not None:
        environ["RUNPOD_API_KEY"] = api_key
    if registry_auth_id is not None:
        environ["RUNPOD_CONTAINER_REGISTRY_AUTH_ID"] = registry_auth_id
    return gate.build_config(_args(*extra), environ)


def _provider_pod(config=None):
    config = config or _config()
    return {
        "id": "pod-exact",
        "name": config.pod_name,
        "image": config.image,
        "interruptible": False,
        "locked": False,
        "dockerEntrypoint": ["python3", "-c"],
        "dockerStartCmd": [gate.POD_EVIDENCE_PROGRAM],
        "desiredStatus": "RUNNING",
        "containerDiskInGb": gate.RUNPOD_CONTAINER_DISK_GB,
        "volumeInGb": 0,
        "networkVolume": None,
        "networkVolumeId": None,
        "ports": [],
        "portMappings": {},
        "publicIp": "203.0.113.10",
        "endpointId": None,
        "templateId": None,
        "containerRegistryAuthId": config.registry_auth_id,
        "lastStartedAt": "2026-07-22T09:59:59Z",
        "costPerHr": "2.39",
        "machineId": "machine-1",
        "gpu": {
            "count": 1,
            "displayName": gate.RUNPOD_GPU_TYPE,
        },
        "machine": {
            "secureCloud": True,
            "gpuTypeId": gate.RUNPOD_GPU_TYPE,
            "gpuDisplayName": gate.RUNPOD_GPU_TYPE,
            "gpuType": {"displayName": gate.RUNPOD_GPU_TYPE},
            "dataCenterId": "US-TX-3",
        },
    }


def _receipt_from_payload(payload, *, pod_id="pod-exact"):
    return {
        "id": pod_id,
        "name": payload["name"],
        "imageName": payload["imageName"],
        "dockerArgs": payload["dockerArgs"],
        "gpuCount": payload["gpuCount"],
        "containerDiskInGb": payload["containerDiskInGb"],
        "volumeInGb": payload["volumeInGb"],
        "ports": payload["ports"],
        "networkVolumeId": payload.get("networkVolumeId"),
        "templateId": payload.get("templateId"),
    }


def _good_evidence(config=None, pod=None):
    config = config or _config()
    pod = pod or _provider_pod(config)
    return {
        "schema": gate.EVIDENCE_SCHEMA,
        "run_id": config.run_id,
        "attempt_id": config.attempt_id,
        "expected": {
            "image": config.image,
            "source_sha": config.source_sha,
            "gpu_type": gate.RUNPOD_GPU_TYPE,
            "gpu_count": 1,
            "cloud_type": "SECURE",
        },
        "runpod": {
            "pod_id": pod["id"],
            "data_center_id": "US-TX-3",
            "gpu_count": "1",
        },
        "started_at": "2026-07-22T10:00:00Z",
        "finished_at": "2026-07-22T10:01:00Z",
        "elapsed_seconds": 60,
        "limits": {
            "test_timeout_seconds": config.worker_timeout_seconds,
            "max_wall_seconds": config.max_wall_seconds,
            "max_hourly_usd": str(config.max_hourly_usd),
            "max_cost_usd": str(config.max_cost_usd),
            "computed_cost_bound_usd": str(config.computed_cost_bound_usd),
        },
        "phsolver_info_command": {
            "command": ["/usr/local/bin/phsolver", "info"],
            "exit_code": 0,
            "timed_out": False,
            "stdout": "{}",
            "stderr": "",
        },
        "phsolver_info": {
            "name": "phsolver",
            "git_sha": config.source_sha,
            "gpu": True,
            "gpu_platform": "ROCm",
            "devices": [{
                "index": 0,
                "name": "AMD Instinct MI300X",
                "vendor": "AMD",
                "arch": "gfx942",
            }],
        },
        "inventory": {
            "command": [
                "/usr/local/bin/phcore_tests",
                "--gtest_filter=GpuEquivalence.*",
                "--gtest_list_tests",
            ],
            "exit_code": 0,
            "timed_out": False,
            "stdout": "GpuEquivalence.\n  First\n  Second\n",
            "stderr": "",
            "discovered": 2,
        },
        "equivalence": {
            "command": ["/usr/local/bin/run-gpu-equivalence"],
            "exit_code": 0,
            "timed_out": False,
            "stdout": (
                "gpu_equivalence_discovered=2\n"
                "[==========] Running 2 tests.\n"
                "[  PASSED  ] 2 tests.\n"
            ),
            "stderr": "",
        },
        "rocm_version": "7.2.0",
        "result": "pass",
    }


def test_live_gate_requires_api_key_from_environment():
    with pytest.raises(gate.ConfigurationError, match="RUNPOD_API_KEY"):
        _config(api_key=None)


def test_dry_run_does_not_require_api_key_or_expose_capability_urls():
    config = _config("--dry-run", api_key=None)
    rendered = gate.redacted_dry_run(config)
    serialized = gate.json.dumps(rendered)

    assert rendered["network_requests"] == 0
    assert PUT_URL not in serialized
    assert GET_URL not in serialized
    assert "RUNPOD_API_KEY" not in serialized


@pytest.mark.parametrize(
    "image",
    [
        "ghcr.io/leapfield/image:rocm",
        "ghcr.io/leapfield/image@sha256:" + "a" * 63,
        "ghcr.io/leapfield/image@sha256:" + "A" * 64,
        "https://ghcr.io/leapfield/image@sha256:" + "a" * 64,
        "ghcr.io/leapfield/image:tag@sha256:" + "a" * 64,
    ],
)
def test_exact_image_validation_rejects_tags_and_invalid_digests(image):
    with pytest.raises(gate.ConfigurationError):
        gate.validate_exact_image(image)


@pytest.mark.parametrize(
    "source_sha",
    ["b" * 39, "b" * 41, "B" * 40, "not-a-source-sha"],
)
def test_source_sha_must_be_exact_lowercase_full_sha(source_sha):
    with pytest.raises(gate.ConfigurationError):
        gate.validate_source_sha(source_sha)


def test_create_request_is_exact_secure_on_demand_mi300x_contract():
    config = _config()
    payload = gate.build_create_payload(
        config,
        terminate_after=TERMINATE_AFTER,
    )
    environment = {
        item["key"]: item["value"]
        for item in payload["env"]
    }

    assert payload["cloudType"] == "SECURE"
    assert payload["computeType"] == "GPU"
    assert payload["deployCost"] == 2.50
    assert payload["gpuTypeId"] == "AMD Instinct MI300X OAM"
    assert payload["gpuCount"] == 1
    assert payload["imageName"] == IMAGE
    assert payload["dockerArgs"].startswith('bash -lc "echo ')
    assert payload["dockerArgs"].endswith(' | base64 -d | python3 -"')
    assert payload["ports"] == ""
    assert payload["startJupyter"] is False
    assert payload["startSsh"] is False
    assert payload["supportPublicIp"] is False
    assert payload["volumeInGb"] == 0
    assert payload["globalNetwork"] is False
    assert payload["terminateAfter"] == "2026-07-22T10:30:00.000Z"
    assert "networkVolumeId" not in payload
    assert environment["PHOTONHUB_EVIDENCE_PUT_URL"] == PUT_URL
    assert "PHOTONHUB_EVIDENCE_GET_URL" not in environment
    assert "containerRegistryAuthId" not in payload


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ports", ["22/tcp"], "inbound port"),
        ("portMappings", {"22": 12345}, "public port mapping"),
        ("containerDiskInGb", 50, "container disk"),
        ("volumeInGb", 20, "persistent volume"),
        (
            "networkVolume",
            {"id": "persistent-network-volume"},
            "persistent network volume",
        ),
        ("endpointId", "endpoint-1", "template or endpoint"),
        ("templateId", "template-1", "template or endpoint"),
        ("locked", True, "lock state"),
        ("containerRegistryAuthId", "unexpected-auth", "registry-auth"),
    ],
)
def test_provider_contract_rejects_network_resource_or_identity_drift(
    field, value, message
):
    config = _config()
    pod = _provider_pod(config)
    pod[field] = value

    with pytest.raises(gate.EvidenceError, match=message):
        gate.validate_provider_pod(pod, config)


def test_provider_record_preserves_no_address_or_capability_values():
    config = _config()
    pod = _provider_pod(config)
    hourly = gate.validate_provider_pod(pod, config)
    record = gate.sanitized_provider_record(
        pod,
        hourly,
        terminate_after=TERMINATE_AFTER,
        requested_hourly_usd=config.max_hourly_usd,
    )
    serialized = gate.json.dumps(record)

    assert set(record) == {
        "pod_id",
        "name",
        "image",
        "desired_status",
        "last_started_at",
        "interruptible",
        "locked",
        "cloud_type",
        "secure_cloud",
        "gpu_type",
        "gpu_count",
        "container_disk_gb",
        "persistent_volume_gb",
        "persistent_network_volume_attached",
        "inbound_ports",
        "public_port_mappings",
        "public_ip_assigned",
        "machine_id",
        "data_center_id",
        "deadman",
        "price_ceiling",
        "hourly_usd",
    }
    assert record["inbound_ports"] == []
    assert record["public_port_mappings"] == {}
    assert record["public_ip_assigned"] is True
    assert record["persistent_network_volume_attached"] is False
    assert record["deadman"]["terminate_after"] == (
        "2026-07-22T10:30:00.000Z"
    )
    assert record["price_ceiling"] == {
        "field": "deployCost",
        "requested_hourly_usd": "2.50",
        "create_acknowledged": True,
    }
    assert pod["publicIp"] not in serialized
    assert "containerRegistryAuthId" not in record


def test_private_registry_auth_is_environment_only_and_redacted():
    registry_auth_id = "clzdaifot0001l90809257ynb"
    config = _config("--dry-run", registry_auth_id=registry_auth_id)
    payload = gate.build_create_payload(
        config,
        terminate_after=TERMINATE_AFTER,
    )
    rendered = gate.json.dumps(gate.redacted_dry_run(config))

    assert payload["containerRegistryAuthId"] == registry_auth_id
    assert registry_auth_id not in rendered
    assert "<redacted-registry-auth-id>" in rendered
    assert not any(
        action.dest == "registry_auth_id"
        for action in gate.build_parser()._actions
    )


@pytest.mark.parametrize(
    "registry_auth_id",
    ["contains spaces", "../credential", "a" * 192],
)
def test_private_registry_auth_id_is_strictly_bounded(registry_auth_id):
    with pytest.raises(gate.ConfigurationError, match="REGISTRY_AUTH_ID"):
        _config(registry_auth_id=registry_auth_id)


def test_reconciliation_lists_inventory_then_filters_exact_name_locally():
    client = gate.RunPodRestClient(
        api_key="provider-secret",
        base_url="https://runpod.example.test/v1",
    )
    config = _config()
    calls = []
    client._request = lambda method, path, **kwargs: (
        calls.append((method, path, kwargs))
        or [
            {"id": "other", "name": config.pod_name + "-other"},
            {"id": "exact", "name": config.pod_name},
        ]
    )

    assert client.list_by_name(config.pod_name) == [
        {"id": "exact", "name": config.pod_name}
    ]
    assert calls == [(
        "GET",
        "/pods",
        {"query": {"includeMachine": "true"}},
    )]


def test_get_pod_requests_machine_and_network_volume_evidence():
    client = gate.RunPodRestClient(
        api_key="provider-secret",
        base_url="https://runpod.example.test/v1",
    )
    calls = []
    client._request = lambda method, path, **kwargs: (
        calls.append((method, path, kwargs))
        or _provider_pod()
    )

    client.get_pod("pod-exact")

    assert calls == [(
        "GET",
        "/pods/pod-exact",
        {
            "query": {
                "includeMachine": "true",
                "includeNetworkVolume": "true",
            }
        },
    )]


def test_create_uses_graphql_deadman_mutation_and_returns_exact_receipt():
    config = _config()
    payload = gate.build_create_payload(
        config,
        terminate_after=TERMINATE_AFTER,
    )
    captured = {}

    class Response:
        status = 200

        def __init__(self, raw):
            self.raw = raw
            self.headers = {"Content-Length": str(len(raw))}

        def read(self, _limit):
            return self.raw

        def close(self):
            return None

    class Opener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["body"] = gate.json.loads(request.data)
            receipt = _receipt_from_payload(
                captured["body"]["variables"]["input"]
            )
            return Response(gate.json.dumps({
                "data": {"podFindAndDeployOnDemand": receipt}
            }).encode())

    client = gate.RunPodRestClient(
        api_key="provider-secret",
        graphql_url="https://graphql.example.test/",
    )
    client._opener = Opener()

    receipt = client.create_pod(payload)

    assert captured["url"] == "https://graphql.example.test/"
    assert captured["body"]["query"] == gate.GRAPHQL_CREATE_MUTATION
    assert captured["body"]["variables"]["input"]["terminateAfter"] == (
        "2026-07-22T10:30:00.000Z"
    )
    assert receipt["id"] == "pod-exact"


def test_create_receipt_must_acknowledge_exact_command_and_pod_shape():
    config = _config()
    payload = gate.build_create_payload(
        config,
        terminate_after=TERMINATE_AFTER,
    )
    receipt = _receipt_from_payload(payload)
    gate.validate_create_receipt(
        receipt,
        config,
        terminate_after=TERMINATE_AFTER,
    )

    receipt["dockerArgs"] = "python3 -c 'unreviewed'"
    with pytest.raises(gate.EvidenceError, match="deadman Pod contract"):
        gate.validate_create_receipt(
            receipt,
            config,
            terminate_after=TERMINATE_AFTER,
        )


def test_provider_contract_requires_explicit_null_network_volume():
    config = _config()
    pod = _provider_pod(config)
    del pod["networkVolume"]

    with pytest.raises(gate.EvidenceError, match="persistent network volume"):
        gate.validate_provider_pod(pod, config)


def test_pod_name_is_deterministic_and_unique_to_evidence_capability_pair():
    first = _config()
    same = _config()
    other = _config(get_url=GET_URL + "-other")

    assert first.pod_name == same.pod_name
    assert first.run_id == same.run_id
    assert first.pod_name != other.pod_name
    assert first.run_id != other.run_id


def test_attempt_id_is_required_and_makes_each_dispatch_unique():
    environ = {
        "RUNPOD_API_KEY": "provider-secret",
        "PHOTONHUB_EVIDENCE_PUT_URL": PUT_URL,
        "PHOTONHUB_EVIDENCE_GET_URL": GET_URL,
    }
    with pytest.raises(gate.ConfigurationError, match="GATE_ATTEMPT_ID"):
        gate.build_config(_args(), environ)

    second_env = dict(environ, PHOTONHUB_GATE_ATTEMPT_ID="test-attempt-2")
    second = gate.build_config(_args(), second_env)
    assert second.pod_name != _config().pod_name
    assert second.run_id != _config().run_id


@pytest.mark.parametrize(
    "extra",
    [
        ("--max-cost-usd", "5.01"),
        (
            "--max-wall-seconds",
            "7200",
            "--max-hourly-usd",
            "2.51",
            "--max-cost-usd",
            "5.00",
        ),
        (
            "--max-wall-seconds",
            "7200",
            "--max-hourly-usd",
            "2.50",
            "--max-cost-usd",
            "5.00",
        ),
        ("--max-wall-seconds", "179"),
        ("--max-wall-seconds", "7201"),
    ],
)
def test_budget_and_wall_bounds_fail_closed(extra):
    with pytest.raises(gate.ConfigurationError):
        _config(*extra)


def test_provider_hourly_price_must_remain_under_declared_ceiling():
    config = _config("--max-hourly-usd", "2.00")
    with pytest.raises(gate.EvidenceError, match="hourly price"):
        gate.validate_provider_pod(_provider_pod(config), config)


def test_ambiguous_create_is_never_retried_and_recovers_exact_name():
    config = _config("--reconcile-attempts", "3")
    recovered = {"id": "pod-recovered", "name": config.pod_name}

    class Client:
        def __init__(self):
            self.list_results = [[], [], [recovered]]
            self.create_calls = 0

        def list_by_name(self, _name):
            return self.list_results.pop(0)

        def create_pod(self, _payload):
            self.create_calls += 1
            raise gate.AmbiguousCreateError("lost response")

    client = Client()
    pod_id, created, receipt = gate.launch_or_reconcile(
        client,
        config,
        terminate_after=TERMINATE_AFTER,
        sleep=lambda _seconds: None,
    )

    assert pod_id == "pod-recovered"
    assert created is True
    assert receipt is None
    assert client.create_calls == 1


def test_cleanup_deletes_late_visible_ambiguous_pod_and_confirms_absence():
    pod_name = "ph-mi300x-gate-late"

    class Client:
        def __init__(self):
            self.list_results = [
                [],
                [{"id": "pod-late", "name": pod_name}],
                [],
                [],
            ]
            self.deleted = []

        def list_by_name(self, _name):
            return self.list_results.pop(0)

        def delete_pod(self, pod_id):
            self.deleted.append(pod_id)

    client = Client()
    deleted = gate.terminate_and_confirm(
        client,
        pod_name=pod_name,
        pod_id=None,
        attempts=5,
        poll_seconds=0,
        sleep=lambda _seconds: None,
    )

    assert client.deleted == ["pod-late"]
    assert deleted == ["pod-late"]


def test_cleanup_does_not_accept_name_absence_after_failed_known_id_delete():
    class Client:
        def list_by_name(self, _name):
            return []

        def delete_pod(self, _pod_id):
            raise gate.ProviderError("temporary failure")

    with pytest.raises(gate.CleanupError):
        gate.terminate_and_confirm(
            Client(),
            pod_name="ph-known",
            pod_id="pod-known",
            attempts=3,
            poll_seconds=0,
            sleep=lambda _seconds: None,
        )


def test_live_orchestration_accepts_only_after_evidence_and_cleanup(tmp_path):
    config = gate.dataclasses.replace(_config(), output_dir=tmp_path)
    pod = _provider_pod(config)
    now = dt.datetime.now(dt.timezone.utc)
    pod["lastStartedAt"] = now.isoformat().replace("+00:00", "Z")
    evidence = _good_evidence(config, pod)
    evidence["started_at"] = now.isoformat().replace("+00:00", "Z")
    evidence["finished_at"] = (now + dt.timedelta(seconds=1)).isoformat().replace(
        "+00:00", "Z"
    )

    class Client:
        def __init__(self):
            self.created = []
            self.deleted = []

        def list_by_name(self, _name):
            return []

        def create_pod(self, payload):
            self.created.append(payload)
            return _receipt_from_payload(payload, pod_id=pod["id"])

        def get_pod(self, _pod_id):
            return pod

        def delete_pod(self, pod_id):
            self.deleted.append(pod_id)

    client = Client()
    result_path = gate.run_live_gate(
        config,
        client=client,
        sleep=lambda _seconds: None,
        evidence_fetcher=lambda _url: evidence,
    )
    report = gate.json.loads(result_path.read_text(encoding="utf-8"))
    lease = gate.json.loads(config.lease_path.read_text(encoding="utf-8"))

    assert len(client.created) == 1
    assert client.deleted == [pod["id"]]
    assert set(report) == {
        "schema",
        "accepted",
        "pod_name",
        "run_id",
        "attempt_id",
        "controller_started_at",
        "requested_image",
        "requested_source_sha",
        "limits",
        "provider",
        "worker",
        "cleanup",
        "error_type",
        "completed_at",
    }
    assert set(report["limits"]) == {
        "max_wall_seconds",
        "max_hourly_usd",
        "max_cost_usd",
        "computed_cost_bound_usd",
        "provider_terminate_after",
    }
    assert report["accepted"] is True
    assert report["cleanup"]["confirmed"] is True
    assert report["provider"]["deadman"]["create_acknowledged"] is True
    assert report["provider"]["price_ceiling"] == {
        "field": "deployCost",
        "requested_hourly_usd": "2.50",
        "create_acknowledged": True,
    }
    assert client.created[0]["deployCost"] == float(
        report["provider"]["price_ceiling"]["requested_hourly_usd"]
    )
    assert report["limits"]["provider_terminate_after"] == (
        report["provider"]["deadman"]["terminate_after"]
    )
    assert client.created[0]["terminateAfter"] == (
        report["provider"]["deadman"]["terminate_after"]
    )
    assert lease["status"] == "cleanup_confirmed"
    assert set(lease) == {
        "schema",
        "pod_name",
        "image",
        "source_sha",
        "attempt_id",
        "created_at",
        "provider_terminate_after",
        "status",
        "pod_id",
        "updated_at",
        "hourly_usd",
        "cleanup_confirmed_at",
        "deleted_pod_ids",
    }
    assert lease["provider_terminate_after"] == (
        report["provider"]["deadman"]["terminate_after"]
    )


def test_live_orchestration_cleans_up_when_provider_contract_fails(tmp_path):
    config = gate.dataclasses.replace(_config(), output_dir=tmp_path)
    pod = _provider_pod(config)
    pod["machine"]["secureCloud"] = False

    class Client:
        def __init__(self):
            self.deleted = []

        def list_by_name(self, _name):
            return []

        def create_pod(self, _payload):
            return _receipt_from_payload(_payload, pod_id=pod["id"])

        def get_pod(self, _pod_id):
            return pod

        def delete_pod(self, pod_id):
            self.deleted.append(pod_id)

    client = Client()
    with pytest.raises(gate.EvidenceError, match="Secure Cloud"):
        gate.run_live_gate(
            config,
            client=client,
            sleep=lambda _seconds: None,
            evidence_fetcher=lambda _url: pytest.fail("must not fetch evidence"),
        )

    assert client.deleted == [pod["id"]]
    lease = gate.json.loads(config.lease_path.read_text(encoding="utf-8"))
    assert lease["status"] == "cleanup_confirmed"


@pytest.mark.parametrize(
    ("path", "bad_value"),
    [
        (("expected", "image"), "ghcr.io/x/y@sha256:" + "c" * 64),
        (("expected", "source_sha"), "c" * 40),
        (("phsolver_info", "git_sha"), "c" * 40),
        (("phsolver_info", "devices", 0, "vendor"), "NVIDIA"),
        (("phsolver_info", "devices", 0, "arch"), "gfx950"),
        (("phsolver_info", "devices", 0, "name"), "AMD Instinct MI325X"),
    ],
)
def test_worker_evidence_rejects_wrong_digest_sha_vendor_arch_or_model(
    path, bad_value
):
    config = _config()
    pod = _provider_pod(config)
    evidence = copy.deepcopy(_good_evidence(config, pod))
    target = evidence
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = bad_value

    with pytest.raises(gate.EvidenceError):
        gate.validate_worker_evidence(evidence, config, pod)


def test_worker_evidence_rejects_a_previous_attempt_id():
    config = _config()
    pod = _provider_pod(config)
    evidence = _good_evidence(config, pod)
    evidence["attempt_id"] = "previous-attempt"

    with pytest.raises(gate.EvidenceError, match="attempt id"):
        gate.validate_worker_evidence(evidence, config, pod)


def test_worker_evidence_is_bound_to_provider_data_center():
    config = _config()
    pod = _provider_pod(config)
    evidence = _good_evidence(config, pod)
    evidence["runpod"]["data_center_id"] = "OTHER-DC"

    with pytest.raises(gate.EvidenceError, match="data center"):
        gate.validate_worker_evidence(evidence, config, pod)


def test_provider_evidence_rejects_non_exact_mi300x_type_and_non_secure_cloud():
    config = _config()
    pod = _provider_pod(config)
    pod["gpu"]["displayName"] = "AMD Instinct MI325X OAM"
    pod["machine"]["gpuTypeId"] = "AMD Instinct MI325X OAM"
    pod["machine"]["gpuDisplayName"] = "AMD Instinct MI325X OAM"
    pod["machine"]["gpuType"]["displayName"] = "AMD Instinct MI325X OAM"
    with pytest.raises(gate.EvidenceError, match="exact MI300X"):
        gate.validate_provider_pod(pod, config)

    pod = _provider_pod(config)
    pod["machine"]["secureCloud"] = False
    with pytest.raises(gate.EvidenceError, match="Secure Cloud"):
        gate.validate_provider_pod(pod, config)

    pod = _provider_pod(config)
    pod["desiredStatus"] = "EXITED"
    with pytest.raises(gate.EvidenceError, match="RUNNING"):
        gate.validate_provider_pod(pod, config)


def test_worker_evidence_requires_every_inventory_test_to_pass_without_skips():
    config = _config()
    pod = _provider_pod(config)
    evidence = _good_evidence(config, pod)
    gate.validate_worker_evidence(evidence, config, pod)

    evidence["equivalence"]["stdout"] += "[  SKIPPED ] 1 test.\n"
    with pytest.raises(gate.EvidenceError, match="skipped"):
        gate.validate_worker_evidence(evidence, config, pod)


def test_worker_evidence_must_be_fresh_for_controller_and_provider_lifecycle():
    config = _config()
    pod = _provider_pod(config)
    evidence = _good_evidence(config, pod)

    gate.validate_worker_evidence(
        evidence,
        config,
        pod,
        not_before=gate.dt.datetime.fromisoformat("2026-07-22T10:00:30+00:00"),
        observed_at=gate.dt.datetime.fromisoformat("2026-07-22T10:02:00+00:00"),
    )
    with pytest.raises(gate.EvidenceError, match="controller attempt"):
        gate.validate_worker_evidence(
            evidence,
            config,
            pod,
            not_before=gate.dt.datetime.fromisoformat(
                "2026-07-22T10:03:00+00:00"
            ),
        )

    pod["lastStartedAt"] = "2026-07-22T10:03:00Z"
    with pytest.raises(gate.EvidenceError, match="provider Pod lifecycle"):
        gate.validate_worker_evidence(evidence, config, pod)


@pytest.mark.parametrize("status,missing", [(404, True), (403, False)])
def test_evidence_poll_retries_only_a_missing_object(monkeypatch, status, missing):
    class FailingOpener:
        def open(self, request, timeout):
            raise gate.urllib.error.HTTPError(
                request.full_url, status, "test", {}, None
            )

    monkeypatch.setattr(
        gate.urllib.request, "build_opener", lambda *handlers: FailingOpener()
    )
    if missing:
        assert gate.fetch_evidence(GET_URL) is None
    else:
        with pytest.raises(gate.EvidenceError, match=r"HTTP 403"):
            gate.fetch_evidence(GET_URL)


def test_manual_workflow_is_pinned_protected_serial_and_fixed_to_five_dollars():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "environment: mi300x-hardware-gate" in workflow
    assert "concurrency:" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "if: github.ref == 'refs/heads/main'" in workflow
    assert "--max-cost-usd 5.00" in workflow
    assert "--max-wall-seconds 1800" in workflow
    assert "--max-hourly-usd 2.50" in workflow
    assert "secrets.RUNPOD_API_KEY" in workflow
    assert "secrets.PHOTONHUB_EVIDENCE_PUT_URL" in workflow
    assert "secrets.PHOTONHUB_EVIDENCE_GET_URL" in workflow
    assert "secrets.RUNPOD_CONTAINER_REGISTRY_AUTH_ID" in workflow
    assert "PHOTONHUB_GATE_ATTEMPT_ID: ${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    assert "actions/checkout@fbc6f3992d24b796d5a048ff273f7fcc4a7b6c09" in workflow
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in workflow
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in workflow
