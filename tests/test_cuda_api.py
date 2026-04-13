import openturbo


def test_cuda_api_surface_is_present():
    assert hasattr(openturbo, "encode_tile_fused")
    assert hasattr(openturbo, "scan_query_many_cache")
    assert hasattr(openturbo, "scan_query_many_cache_multi_tile")
    assert hasattr(openturbo, "is_cuda_extension_available")


def test_require_cuda_extension_matches_import_state():
    if openturbo.is_cuda_extension_available():
        openturbo.require_cuda_extension()
        return

    try:
        openturbo.require_cuda_extension()
    except RuntimeError as exc:
        assert "not built yet" in str(exc)
    else:
        raise AssertionError("require_cuda_extension should fail when the extension is unavailable")
