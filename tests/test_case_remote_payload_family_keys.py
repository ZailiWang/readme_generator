#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.tools.memory_tool import GlobalMemory  # noqa: E402
from readme_generator.tools.remote_exec_tool import RemoteExecutionTool  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        mem = GlobalMemory(str(Path(td) / "mem.json"))
        RemoteExecutionTool.global_memory = mem
        mem.memory_store("family_md", "# Llama 3.2")
        mem.memory_store("family_index_js", "const name = 'Llama32ConfigGenerator';")
        mem.memory_store("input_text", '{"model_list":["Llama-3.2-3B-Instruct"]}')

        resolved = RemoteExecutionTool._resolve_legacy_content({})
        assert "family_md" in resolved and "family_index_js" in resolved
        assert "ref_md" not in resolved and "ref_index_js" not in resolved
        assert "Llama 3.2" in resolved["family_md"]
        assert "Llama32" in resolved["family_index_js"]

    print("PASS: remote payload uses family_md/family_index_js keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
