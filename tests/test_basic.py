import openturbo


def test_version_exists():
    assert hasattr(openturbo, "__version__")
    assert openturbo.__version__ == "0.0.1"
