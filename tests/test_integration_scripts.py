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
    assert "${TARGET}" in cmake_text
    assert "openturbo_shadow_eval_callback.hpp" in source_text
    assert "params.cb_eval = openturbo_shadow_cb_eval;" in source_text
    assert source_text.count("openturbo_shadow_eval_callback.hpp") == 1


def test_parse_probe_output_requires_both_lines() -> None:
    output = "\n".join(
        [
            "[openturbo] cpy_k probe layer=0 compatible=1 head_dim=128",
            "[openturbo] shadow_encode layer=0 status=success num_tiles=16",
        ]
    )

    probe_line, shadow_line = probe_runner.parse_probe_output(output, 0)

    assert probe_line.startswith("[openturbo] cpy_k probe")
    assert shadow_line.startswith("[openturbo] shadow_encode")


def test_parse_probe_output_rejects_missing_shadow_line() -> None:
    with pytest.raises(RuntimeError, match="shadow_encode"):
        probe_runner.parse_probe_output("[openturbo] cpy_k probe layer=0 compatible=1", 0)


def test_write_text_file_refuses_overwrite_without_force(tmp_path: Path) -> None:
    target = tmp_path / "sample.txt"
    scaffold.write_text_file(target, "first", force=True)

    with pytest.raises(FileExistsError):
        scaffold.write_text_file(target, "second", force=False)