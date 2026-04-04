"""Import smoke tests for core packages (no heavy I/O)."""


def test_import_admm():
    from algos.admm_pnp import ADMMPnP

    assert ADMMPnP is not None


def test_import_unet():
    from models.unet import create_model

    assert callable(create_model)


def test_import_inference_service():
    from inference.service import SARDenoiseService

    assert SARDenoiseService is not None


def test_import_api_main():
    import pytest

    pytest.importorskip("fastapi")
    import api.main

    assert hasattr(api.main, "app")
