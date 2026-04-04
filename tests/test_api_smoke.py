"""FastAPI smoke tests (TestClient)."""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.deps import reset_service_for_tests
from api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("SAR_CHECKPOINT", raising=False)
    reset_service_for_tests()
    with TestClient(app) as c:
        yield c
    reset_service_for_tests()


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert data.get("version") == "0.1.0"
    assert data.get("direct_infer_backend") in ("pytorch", "onnx")
    assert data.get("onnx_path_set") in (True, False)
    if "git_sha" in data:
        assert isinstance(data["git_sha"], str) and len(data["git_sha"]) >= 4


def test_health_direct_backend_reflects_sar_backend(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_BACKEND", "onnx")
    monkeypatch.setenv("SAR_ONNX_PATH", "/nonexistent/model.onnx")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("direct_infer_backend") == "onnx"
    assert r.json().get("onnx_path_set") is True


def test_optional_api_key_blocks_v1_without_header(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_API_KEY", "integration-test-secret")
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    r = client.post(
        "/v1/denoise",
        params={"method": "TV Denoising"},
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 401
    assert "API key" in r.json().get("detail", "")


def test_optional_api_key_accepts_bearer(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_API_KEY", "integration-test-secret")
    buf = io.BytesIO()
    Image.fromarray((np.random.RandomState(2).rand(16, 16) * 255).astype(np.uint8)).save(
        buf, format="PNG"
    )
    buf.seek(0)
    r = client.post(
        "/v1/denoise",
        params={"method": "TV Denoising"},
        files={"file": ("x.png", buf.getvalue(), "image/png")},
        headers={"Authorization": "Bearer integration-test-secret"},
    )
    assert r.status_code == 200, r.text


def test_health_ok_when_api_key_configured(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_API_KEY", "integration-test-secret")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_ready_queue_disabled(client: TestClient):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "queue": "disabled"}


def test_ready_queue_no_redis_url(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_USE_QUEUE", "1")
    monkeypatch.delenv("REDIS_URL", raising=False)
    r = client.get("/ready")
    assert r.status_code == 503
    assert r.json().get("status") == "not_ready"


def test_ready_queue_redis_ok(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_USE_QUEUE", "1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    inst = MagicMock()
    inst.ping.return_value = True
    with patch("redis.Redis") as MockRedis:
        MockRedis.from_url.return_value = inst
        r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "queue": "redis"}


def test_ready_queue_redis_ping_fails(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_USE_QUEUE", "1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    with patch("redis.Redis") as MockRedis:
        MockRedis.from_url.side_effect = OSError("connection refused")
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json().get("status") == "not_ready"


def test_ready_ok_when_api_key_configured(monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_API_KEY", "integration-test-secret")
    r = client.get("/ready")
    assert r.status_code == 200


def test_json_access_log_and_request_id(caplog, monkeypatch, client: TestClient):
    monkeypatch.setenv("SAR_ACCESS_LOG_JSON", "1")
    with caplog.at_level(logging.INFO, logger="sar.api.access"):
        r = client.get("/health", headers={"X-Request-ID": "trace-abc"})
    assert r.status_code == 200
    assert r.headers.get("X-Request-ID") == "trace-abc"
    access_recs = [r for r in caplog.records if r.name == "sar.api.access"]
    assert access_recs
    payload = json.loads(access_recs[-1].getMessage())
    assert payload["event"] == "http_request"
    assert payload["path"] == "/health"
    assert payload["request_id"] == "trace-abc"


def test_denoise_tv_returns_png(client: TestClient):
    buf = io.BytesIO()
    Image.fromarray((np.random.RandomState(0).rand(24, 32) * 255).astype(np.uint8)).save(
        buf, format="PNG"
    )
    buf.seek(0)
    r = client.post(
        "/v1/denoise",
        params={"method": "TV Denoising"},
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.headers.get("content-type") == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (32, 24)


def test_invalid_image(client: TestClient):
    r = client.post(
        "/v1/denoise",
        params={"method": "TV Denoising"},
        files={"file": ("x.bin", b"not a png", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_denoise_json_direct_with_uncertainty(client: TestClient):
    buf = io.BytesIO()
    Image.fromarray((np.random.RandomState(1).rand(64, 64) * 255).astype(np.uint8)).save(
        buf, format="PNG"
    )
    buf.seek(0)
    r = client.post(
        "/v1/denoise",
        params={
            "method": "Direct Denoising",
            "response_format": "json",
            "include_uncertainty": True,
            "uncertainty_tta_passes": 2,
        },
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "denoised_png" in data and "uncertainty_png" in data
    assert "meta" in data and "inference_ms" in data["meta"]
    assert "uncertainty_mean" in data["meta"]


def test_denoise_json_include_blind_qa(client: TestClient):
    buf = io.BytesIO()
    Image.fromarray((np.random.RandomState(3).rand(48, 48) * 255).astype(np.uint8)).save(
        buf, format="PNG"
    )
    buf.seek(0)
    r = client.post(
        "/v1/denoise",
        params={
            "method": "TV Denoising",
            "response_format": "json",
            "include_blind_qa": True,
        },
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200, r.text
    meta = r.json().get("meta") or {}
    assert "blind_qa" in meta
    bq = meta["blind_qa"]
    assert "enl_homogeneous_median" in bq
    assert "edge_preservation_vs_input" in bq


def test_include_uncertainty_requires_json(client: TestClient):
    buf = io.BytesIO()
    Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    r = client.post(
        "/v1/denoise",
        params={
            "method": "Direct Denoising",
            "include_uncertainty": True,
        },
        files={"file": ("x.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 400
