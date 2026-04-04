"""Unit tests for api.storage (no Redis)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def jobs_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SAR_JOBS_DIR", str(tmp_path / "jobs"))
    yield tmp_path / "jobs"


def test_job_roundtrip(jobs_root):
    from api import storage as s

    jid = s.new_job_id()
    d = s.ensure_job_dir(jid)
    assert d.is_dir()
    s.write_meta(jid, {"method": "TV Denoising", "rq_job_id": "x"})
    s.write_status(jid, "queued")
    assert s.read_meta(jid)["method"] == "TV Denoising"
    assert s.read_status(jid)["status"] == "queued"
    s.write_status(jid, "done")
    assert s.read_status(jid)["status"] == "done"
