from app import __version__


def test_project_version() -> None:
    """Verify that the application package exposes its version."""
    assert __version__ == "0.1.0"