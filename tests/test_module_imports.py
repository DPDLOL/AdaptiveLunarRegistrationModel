"""Module-level import checks for the registration package."""

from __future__ import annotations


def test_preprocessing_module_imports() -> None:
    from backend.registration import preprocessing

    assert preprocessing is not None


def test_orb_seed_module_imports() -> None:
    from backend.registration import orb_seed

    assert orb_seed is not None


def test_akaze_recovery_module_imports() -> None:
    from backend.registration import akaze_recovery

    assert akaze_recovery is not None


def test_optical_flow_module_imports() -> None:
    from backend.registration import optical_flow

    assert optical_flow is not None


def test_geometry_module_imports() -> None:
    from backend.registration import geometry

    assert geometry is not None


def test_rotation_module_imports() -> None:
    from backend.registration import rotation

    assert rotation is not None


def test_validation_module_imports() -> None:
    from backend.registration import validation

    assert validation is not None


def test_visualization_module_imports() -> None:
    from backend.registration import visualization

    assert visualization is not None


def test_pipeline_module_imports() -> None:
    from backend.registration import pipeline

    assert pipeline is not None
