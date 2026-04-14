from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))


def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe_runner = load_module("run_llama_cpp_k_cache_probe", SCRIPTS_ROOT / "run_llama_cpp_k_cache_probe.py")
scaffold = load_module("scaffold_llama_cpp_integration", SCRIPTS_ROOT / "scaffold_llama_cpp_integration.py")


SRC_CMAKELISTS_TEMPLATE = """llama_add_compile_flags()

target_link_libraries(llama PUBLIC ggml)
"""


KV_CACHE_TEMPLATE = """#include \"llama-model.h\"
#include <cmath>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

void demo() {
    const int64_t n_embd_gqa = n_embd_head * n_head;
}
"""


EVAL_CALLBACK_CMAKELISTS_TEMPLATE = """set(TARGET llama-eval-callback)
add_executable(${TARGET} eval-callback.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
"""


EVAL_CALLBACK_SOURCE_TEMPLATE = """#include \"arg.h\"
#include \"common.h\"
#include \"debug.h\"
#include \"log.h\"
#include \"llama.h\"
#include \"llama-cpp.h\"

int main(int argc, char ** argv) {
    base_callback_data cb_data;

    common_params params;

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    params.cb_eval = common_debug_cb_eval<false>;
    params.cb_eval_user_data = &cb_data;
    params.warmup = false;

    return 0;
}
"""


def write_llama_probe_tree(root: Path) -> tuple[Path, Path, Path, Path]:
    src_root = root / "src"
    eval_root = root / "examples" / "eval-callback"
    output_root = root / "examples" / "openturbo"
    src_root.mkdir(parents=True)
    eval_root.mkdir(parents=True)
    output_root.mkdir(parents=True)

    src_cmakelists = src_root / "CMakeLists.txt"
    kv_cache = src_root / "llama-kv-cache.cpp"
    eval_cmakelists = eval_root / "CMakeLists.txt"
    eval_source = eval_root / "eval-callback.cpp"

    src_cmakelists.write_text(SRC_CMAKELISTS_TEMPLATE, encoding="utf-8")
    kv_cache.write_text(KV_CACHE_TEMPLATE, encoding="utf-8")
    eval_cmakelists.write_text(EVAL_CALLBACK_CMAKELISTS_TEMPLATE, encoding="utf-8")
    eval_source.write_text(EVAL_CALLBACK_SOURCE_TEMPLATE, encoding="utf-8")
    return output_root, src_cmakelists, kv_cache, eval_source


def test_generate_scaffold_writes_shadow_and_bridge_files(tmp_path: Path) -> None:
    output_root = tmp_path / "generated"

    probe_runner.generate_scaffold(output_root, force=True)

    assert (output_root / "openturbo_llama_cpp_bridge.hpp").exists()
    assert (output_root / "openturbo_llama_cpp_bridge.cpp").exists()
    assert (output_root / "openturbo_shadow_eval_callback.hpp").exists()
    assert (output_root / "openturbo_shadow_eval_callback.cpp").exists()

    shadow_text = (output_root / "openturbo_shadow_eval_callback.cpp").read_text(encoding="utf-8")
    assert "attn_inp_kq_mask" in shadow_text
    assert "openturbo_compute_mask_read_coverage" in shadow_text
    assert "shadow_score" in shadow_text
    assert "shadow_compare" in shadow_text
    assert "shadow_snr" in shadow_text
    assert "shadow_components" in shadow_text
    assert "shadow_legacy" in shadow_text
    assert "corner_max_abs_error=%.6f\\n" in shadow_text
    assert "shadow_legacy layer=%d status=%s node=%s\\n" in shadow_text
    assert "query_head_index / query_head_group" in shadow_text
    assert "OPENTURBO_SHADOW_LAYER_FILTER" in shadow_text
    assert "shadow_packed_path" in shadow_text


def test_apply_probe_patch_is_idempotent(tmp_path: Path) -> None:
    output_root, src_cmakelists, kv_cache, _ = write_llama_probe_tree(tmp_path)

    changed = scaffold.apply_probe_patch(tmp_path, output_root)
    unchanged = scaffold.apply_probe_patch(tmp_path, output_root)

    assert changed is True
    assert unchanged is False
    cmake_text = src_cmakelists.read_text(encoding="utf-8")
    kv_text = kv_cache.read_text(encoding="utf-8")
    assert "OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE" in cmake_text
    assert cmake_text.count("OPENTURBO_EXPERIMENTAL_K_CACHE_PROBE_ONLY") == 1
    assert "openturbo_probe_k_cache_write" in kv_text
    assert kv_text.count("openturbo_probe_k_cache_write(k_cur, k, il, get_size(), k->ne[2]);") == 1


def test_apply_shadow_encode_patch_is_idempotent_and_preserves_cmake_vars(tmp_path: Path) -> None:
    output_root, _, _, eval_source = write_llama_probe_tree(tmp_path)
    eval_cmakelists = tmp_path / "examples" / "eval-callback" / "CMakeLists.txt"

    changed = scaffold.apply_shadow_encode_patch(tmp_path, output_root)
    unchanged = scaffold.apply_shadow_encode_patch(tmp_path, output_root)

    assert changed is True
    assert unchanged is False

    cmake_text = eval_cmakelists.read_text(encoding="utf-8")
    source_text = eval_source.read_text(encoding="utf-8")
    assert "OPENTURBO_EXPERIMENTAL_K_CACHE_SHADOW_ENCODE" in cmake_text
    assert "OPENTURBO_EXPERIMENTAL_PACKED_SCORE_PATH" in cmake_text
    assert "${TARGET}" in cmake_text
    assert "openturbo_shadow_eval_callback.hpp" in source_text
    assert "params.cb_eval = openturbo_shadow_cb_eval;" in source_text
    assert source_text.count("openturbo_shadow_eval_callback.hpp") == 1


def test_apply_packed_score_patch_invokes_tracked_patch_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ggml_cuda_root = tmp_path / "ggml" / "src" / "ggml-cuda"
    ggml_cuda_root.mkdir(parents=True)
    (ggml_cuda_root / "fattn.cu").write_text("void placeholder() {}\n", encoding="utf-8")
    (ggml_cuda_root / "ggml-cuda.cu").write_text("void placeholder_dispatch() {}\n", encoding="utf-8")
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "llama-kv-cache.cpp").write_text("void placeholder_kv_cache() {}\n", encoding="utf-8")

    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()
    patch_files = [patch_dir / "sidecar_core.patch", patch_dir / "fattn_injection.patch", patch_dir / "llama_hooks.patch"]
    for patch_file in patch_files:
        patch_file.write_text("diff --git a/foo b/foo\n", encoding="utf-8")

    recorded: list[list[str]] = []

    def fake_run(command: list[str], check: bool) -> None:
        recorded.append(command)

    monkeypatch.setattr(scaffold.subprocess, "run", fake_run)

    changed = scaffold.apply_packed_score_patch(tmp_path, patch_dir)

    assert changed is True
    assert recorded == [
        ["git", "-C", str(tmp_path), "apply", "--whitespace=nowarn", str(path.resolve())]
        for path in patch_files
    ]


def test_apply_packed_score_patch_is_idempotent_when_markers_exist(tmp_path: Path) -> None:
    ggml_cuda_root = tmp_path / "ggml" / "src" / "ggml-cuda"
    ggml_cuda_root.mkdir(parents=True)
    (ggml_cuda_root / "openturbo-sidecar.cuh").write_text("#pragma once\n", encoding="utf-8")
    (ggml_cuda_root / "openturbo-sidecar.cu").write_text("void sidecar() {}\n", encoding="utf-8")
    (ggml_cuda_root / "fattn.cu").write_text("bool ggml_cuda_flash_attn_ext_openturbo_supported(const ggml_tensor * dst) { return dst != nullptr; }\n", encoding="utf-8")
    (ggml_cuda_root / "ggml-cuda.cu").write_text('GGML_LOG_INFO("%s: OpenTurbo packed flash-attn path active\\n", __func__);\n', encoding="utf-8")
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "llama-kv-cache.cpp").write_text("bool openturbo_sync_k_sidecars() { return true; }\n", encoding="utf-8")

    changed = scaffold.apply_packed_score_patch(tmp_path, tmp_path / "missing.patch")

    assert changed is False


def test_resolve_packed_score_patch_files_returns_expected_order(tmp_path: Path) -> None:
    patch_dir = tmp_path / "patches"
    patch_dir.mkdir()

    resolved = scaffold.resolve_packed_score_patch_files(patch_dir)

    assert [path.name for path in resolved] == ["sidecar_core.patch", "fattn_injection.patch", "llama_hooks.patch"]


def test_parse_probe_output_requires_both_lines() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
            "[openturbo] shadow_read layer=0 status=success node=kq-0 expected_rows=1 present_rows=1 tiles_per_row=8",
            "[openturbo] shadow_score layer=0 status=success node=kq-0 active_rows=1 num_heads=8 num_query_tiles=1 first_row=0 first_score=1.0 top_row=0 top_score=1.0",
            "[openturbo] shadow_compare layer=0 status=success node=kq-0 active_rows=1 top_match=1 first_row=0 shadow_first=1.0 dense_first=1.1 shadow_top_row=0 shadow_top=1.0 dense_top_row=0 dense_top=1.1 mae=0.1 max_abs_error=0.1",
            "[openturbo] shadow_snr layer=0 status=success node=kq-0 fwht_signal_power=4.0 fwht_noise_power=0.04 fwht_snr_db=20.0 signal_retention=99.009903 legacy_noise_power=0.16 legacy_snr_db=13.979400 legacy_signal_retention=96.153847",
            "[openturbo] shadow_components layer=0 status=success node=kq-0 first_main=0.8 first_residual=0.2 mean_main=0.8 mean_residual=0.2",
            "[openturbo] shadow_legacy layer=0 status=success node=kq-0 corner_top_match=1 corner_first=0.6 corner_top_row=0 corner_top=0.6 corner_mae=0.2 corner_max_abs_error=0.2",
        ]
    )

    probe_line, shadow_line, shadow_read_line, shadow_score_line, shadow_compare_line, shadow_snr_line, shadow_components_line, shadow_legacy_line = probe_runner.parse_probe_output(output, 0)

    assert probe_line.startswith("[openturbo] cpy_k probe")
    assert shadow_line.startswith("[openturbo] shadow_encode")
    assert shadow_read_line.startswith("[openturbo] shadow_read")
    assert shadow_score_line.startswith("[openturbo] shadow_score")
    assert shadow_compare_line.startswith("[openturbo] shadow_compare")
    assert shadow_snr_line.startswith("[openturbo] shadow_snr")
    assert shadow_components_line.startswith("[openturbo] shadow_components")
    assert shadow_legacy_line.startswith("[openturbo] shadow_legacy")


def test_parse_probe_output_rejects_missing_shadow_line() -> None:
    with pytest.raises(RuntimeError, match="shadow_encode"):
        probe_runner.parse_probe_output("[openturbo] cpy_k probe layer=0 compatible=1", 0)


def test_parse_probe_output_rejects_missing_shadow_read_line() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
        ]
    )

    with pytest.raises(RuntimeError, match="shadow_read"):
        probe_runner.parse_probe_output(output, 0)


def test_parse_probe_output_rejects_missing_shadow_score_line() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
            "[openturbo] shadow_read layer=0 status=success node=kq-0 expected_rows=1 present_rows=1 tiles_per_row=8",
        ]
    )

    with pytest.raises(RuntimeError, match="shadow_score"):
        probe_runner.parse_probe_output(output, 0)


def test_parse_probe_output_rejects_missing_shadow_compare_line() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
            "[openturbo] shadow_read layer=0 status=success node=kq-0 expected_rows=1 present_rows=1 tiles_per_row=8",
            "[openturbo] shadow_score layer=0 status=success node=kq-0 active_rows=1 num_heads=8 num_query_tiles=1 first_row=0 first_score=1.0 top_row=0 top_score=1.0",
        ]
    )

    with pytest.raises(RuntimeError, match="shadow_compare"):
        probe_runner.parse_probe_output(output, 0)


def test_parse_probe_output_rejects_missing_shadow_components_line() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
            "[openturbo] shadow_read layer=0 status=success node=kq-0 expected_rows=1 present_rows=1 tiles_per_row=8",
            "[openturbo] shadow_score layer=0 status=success node=kq-0 active_rows=1 num_heads=8 num_query_tiles=1 first_row=0 first_score=1.0 top_row=0 top_score=1.0",
            "[openturbo] shadow_compare layer=0 status=success node=kq-0 active_rows=1 top_match=1 first_row=0 shadow_first=1.0 dense_first=1.1 shadow_top_row=0 shadow_top=1.0 dense_top_row=0 dense_top=1.1 mae=0.1 max_abs_error=0.1",
            "[openturbo] shadow_snr layer=0 status=success node=kq-0 fwht_signal_power=4.0 fwht_noise_power=0.04 fwht_snr_db=20.0 signal_retention=99.009903 legacy_noise_power=0.16 legacy_snr_db=13.979400 legacy_signal_retention=96.153847",
        ]
    )

    with pytest.raises(RuntimeError, match="shadow_components"):
        probe_runner.parse_probe_output(output, 0)


def test_parse_probe_output_rejects_missing_shadow_snr_line() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
            "[openturbo] shadow_read layer=0 status=success node=kq-0 expected_rows=1 present_rows=1 tiles_per_row=8",
            "[openturbo] shadow_score layer=0 status=success node=kq-0 active_rows=1 num_heads=8 num_query_tiles=1 first_row=0 first_score=1.0 top_row=0 top_score=1.0",
            "[openturbo] shadow_compare layer=0 status=success node=kq-0 active_rows=1 top_match=1 first_row=0 shadow_first=1.0 dense_first=1.1 shadow_top_row=0 shadow_top=1.0 dense_top_row=0 dense_top=1.1 mae=0.1 max_abs_error=0.1",
            "[openturbo] shadow_components layer=0 status=success node=kq-0 first_main=0.8 first_residual=0.2 mean_main=0.8 mean_residual=0.2",
            "[openturbo] shadow_legacy layer=0 status=success node=kq-0 corner_top_match=1 corner_first=0.6 corner_top_row=0 corner_top=0.6 corner_mae=0.2 corner_max_abs_error=0.2",
        ]
    )

    with pytest.raises(RuntimeError, match="shadow_snr"):
        probe_runner.parse_probe_output(output, 0)


def test_parse_probe_output_rejects_missing_shadow_legacy_line() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
            "[openturbo] shadow_read layer=0 status=success node=kq-0 expected_rows=1 present_rows=1 tiles_per_row=8",
            "[openturbo] shadow_score layer=0 status=success node=kq-0 active_rows=1 num_heads=8 num_query_tiles=1 first_row=0 first_score=1.0 top_row=0 top_score=1.0",
            "[openturbo] shadow_compare layer=0 status=success node=kq-0 active_rows=1 top_match=1 first_row=0 shadow_first=1.0 dense_first=1.1 shadow_top_row=0 shadow_top=1.0 dense_top_row=0 dense_top=1.1 mae=0.1 max_abs_error=0.1",
            "[openturbo] shadow_snr layer=0 status=success node=kq-0 fwht_signal_power=4.0 fwht_noise_power=0.04 fwht_snr_db=20.0 signal_retention=99.009903 legacy_noise_power=0.16 legacy_snr_db=13.979400 legacy_signal_retention=96.153847",
            "[openturbo] shadow_components layer=0 status=success node=kq-0 first_main=0.8 first_residual=0.2 mean_main=0.8 mean_residual=0.2",
        ]
    )

    with pytest.raises(RuntimeError, match="shadow_legacy"):
        probe_runner.parse_probe_output(output, 0)


def test_write_text_file_refuses_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    scaffold.write_text_file(target, "first", force=True)

    with pytest.raises(FileExistsError):
        scaffold.write_text_file(target, "second", force=False)


def test_collect_benchmark_rows_merges_compare_and_snr() -> None:
    output = "\n".join(
        [
            "[openturbo] shadow_compare layer=8 status=success node=kq-0 active_rows=2 raw_top_match=1 fwht_top_match=1 first_row=0 shadow_first=1.0 raw_first=0.1 fwht_first=0.9 shadow_top_row=0 shadow_top=1.0 raw_top_row=0 raw_top=0.1 fwht_top_row=0 fwht_top=0.9 raw_mae=0.2 raw_max_abs_error=0.3 fwht_mae=0.05 fwht_max_abs_error=0.07 fwht_mean_scale_ratio=1.02",
            "[openturbo] shadow_snr layer=8 status=success node=kq-0 fwht_signal_power=4.0 fwht_noise_power=0.04 fwht_snr_db=20.0 signal_retention=99.009903 legacy_noise_power=0.16 legacy_snr_db=13.979400 legacy_signal_retention=96.153847",
            "[openturbo] shadow_compare layer=16 status=success node=kq-1 active_rows=2 raw_top_match=1 fwht_top_match=1 first_row=0 shadow_first=1.1 raw_first=0.2 fwht_first=1.0 shadow_top_row=0 shadow_top=1.1 raw_top_row=0 raw_top=0.2 fwht_top_row=0 fwht_top=1.0 raw_mae=0.3 raw_max_abs_error=0.4 fwht_mae=0.08 fwht_max_abs_error=0.1 fwht_mean_scale_ratio=0.98",
            "[openturbo] shadow_snr layer=16 status=success node=kq-1 fwht_signal_power=5.0 fwht_noise_power=0.05 fwht_snr_db=20.0 signal_retention=99.009903 legacy_noise_power=0.25 legacy_snr_db=13.010300 legacy_signal_retention=95.238098",
        ]
    )

    rows = probe_runner.collect_benchmark_rows(output, "P1")

    assert len(rows) == 2
    assert rows[0]["prompt"] == "P1"
    assert rows[0]["layer"] == 8
    assert rows[0]["fwht_mae"] == pytest.approx(0.05)
    assert rows[0]["signal_retention"] == pytest.approx(99.009903)
    assert rows[1]["layer"] == 16


def test_format_benchmark_report_contains_summary() -> None:
    report = probe_runner.format_benchmark_report(
        [
            {"prompt": "P1", "layer": 0, "fwht_mae": 100.0, "fwht_mean_scale_ratio": 1.01, "fwht_top_match": 1, "signal_retention": 99.5, "fwht_snr_db": 23.0},
            {"prompt": "P2", "layer": 8, "fwht_mae": 200.0, "fwht_mean_scale_ratio": 0.99, "fwht_top_match": 1, "signal_retention": 99.0, "fwht_snr_db": 22.0},
        ],
        "0,8",
    )

    assert "Overall average retention" in report
    assert "| P1 | 0 | 99.50 | 23.00 | 100.00 | 1.010 | 1 |" in report


def test_existing_bindings_healthy_returns_true_for_importable_module(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0
        stdout = "A:/OpenTurbo/.venv/Lib/site-packages/openturbo/_openturbo_cuda.cp313-win_amd64.pyd\n"

    monkeypatch.setattr(probe_runner.subprocess, "run", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(probe_runner.Path, "exists", lambda self: True)

    assert probe_runner.existing_bindings_healthy(REPO_ROOT) is True


def test_existing_bindings_healthy_returns_false_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(probe_runner.subprocess, "run", lambda *args, **kwargs: Completed())

    assert probe_runner.existing_bindings_healthy(REPO_ROOT) is False


def test_probe_runner_parser_exposes_force_bindings_refresh_flag() -> None:
    args = probe_runner.build_arg_parser().parse_args(["--force-bindings-refresh"])

    assert args.force_bindings_refresh is True


def test_probe_runner_parser_keeps_packed_score_path_enabled_by_default() -> None:
    args = probe_runner.build_arg_parser().parse_args([])

    assert args.packed_score_path is True