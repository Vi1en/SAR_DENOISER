"""YAML + env merge for API inference defaults."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def isolated_infer_config(monkeypatch, tmp_path: Path):
    import importlib

    y = tmp_path / "infer.yaml"
    y.write_text(
        "device: cuda\nmodel_type: dncnn\ncheckpoint: my.ckpt\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SAR_INFER_CONFIG", str(y))
    import api.infer_config as ic

    importlib.reload(ic)
    yield ic
    monkeypatch.delenv("SAR_INFER_CONFIG", raising=False)
    importlib.reload(ic)


def test_yaml_defaults(isolated_infer_config):
    ic = isolated_infer_config
    m = ic.get_merged()
    assert m["device"] == "cuda"
    assert m["model_type"] == "dncnn"
    assert m["checkpoint"] == "my.ckpt"
    assert ic.model_type_label() == "DnCNN"


def test_env_overrides_yaml(isolated_infer_config, monkeypatch):
    monkeypatch.setenv("SAR_DEVICE", "cpu")
    monkeypatch.setenv("SAR_MODEL_TYPE", "res_unet")
    monkeypatch.setenv("SAR_CHECKPOINT", "env.pth")
    ic = isolated_infer_config
    m = ic.get_merged()
    assert m["device"] == "cpu"
    assert m["model_type"] == "resunet"
    assert m["checkpoint"] == "env.pth"
    assert ic.model_type_label() == "Res-U-Net"
    assert ic.effective_checkpoint_str() == "env.pth"


def test_effective_checkpoint_none_when_unset(monkeypatch):
    import importlib

    monkeypatch.delenv("SAR_CHECKPOINT", raising=False)
    monkeypatch.delenv("SAR_INFER_CONFIG", raising=False)
    import api.infer_config as ic

    importlib.reload(ic)
    assert ic.effective_checkpoint_str() is None


def test_yaml_backend_and_onnx_path(monkeypatch, tmp_path: Path):
    import importlib

    y = tmp_path / "infer.yaml"
    y.write_text(
        "device: cpu\nbackend: ONNX\nonnx_path: models/x.onnx\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SAR_INFER_CONFIG", str(y))
    monkeypatch.delenv("SAR_BACKEND", raising=False)
    monkeypatch.delenv("SAR_ONNX_PATH", raising=False)
    import api.infer_config as ic

    importlib.reload(ic)
    try:
        m = ic.get_merged()
        assert m["device"] == "cpu"
        assert m["backend"] == "onnx"
        assert m["onnx_path"] == "models/x.onnx"
    finally:
        monkeypatch.delenv("SAR_INFER_CONFIG", raising=False)
        importlib.reload(ic)


def test_service_options_from_merged(monkeypatch, tmp_path: Path):
    import importlib

    y = tmp_path / "infer.yaml"
    y.write_text("backend: onnx\nonnx_path: z.onnx\n", encoding="utf-8")
    monkeypatch.setenv("SAR_INFER_CONFIG", str(y))
    import api.infer_config as ic

    importlib.reload(ic)
    try:
        o = ic.service_options_from_merged()
        assert o == {"infer_backend": "onnx", "onnx_path": "z.onnx"}
    finally:
        monkeypatch.delenv("SAR_INFER_CONFIG", raising=False)
        importlib.reload(ic)


def test_env_overrides_backend_onnx_path(monkeypatch, tmp_path: Path):
    import importlib

    y = tmp_path / "infer.yaml"
    y.write_text("backend: onnx\nonnx_path: from.yaml\n", encoding="utf-8")
    monkeypatch.setenv("SAR_INFER_CONFIG", str(y))
    monkeypatch.setenv("SAR_BACKEND", "pytorch")
    monkeypatch.setenv("SAR_ONNX_PATH", "/abs/model.onnx")
    import api.infer_config as ic

    importlib.reload(ic)
    try:
        m = ic.get_merged()
        assert m["backend"] == "pytorch"
        assert m["onnx_path"] == "/abs/model.onnx"
    finally:
        monkeypatch.delenv("SAR_INFER_CONFIG", raising=False)
        monkeypatch.delenv("SAR_BACKEND", raising=False)
        monkeypatch.delenv("SAR_ONNX_PATH", raising=False)
        importlib.reload(ic)

