#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.tools.generate_readme_tool import GenerateReadmeTool  # noqa: E402
from readme_generator.tools.memory_tool import GlobalMemory  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        mem = GlobalMemory(str(Path(td) / "mem.json"))
        GenerateReadmeTool.global_memory = mem
        mem.memory_store("generation_mode", "reference")
        mem.memory_store("model_list", ["Qwen3-8B", "Qwen3-8B"])
        mem.memory_store("github_url", ["", "", ""])
        mem.memory_store("ref_md", "# Reference\n\nRun model deployment.")
        mem.memory_store("ref_index_js", "")

        # Force fallback path to keep test deterministic/offline.
        old_invoke = GenerateReadmeTool.llm.invoke
        GenerateReadmeTool.llm.invoke = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("skip llm"))
        try:
            result = GenerateReadmeTool.memory_generate_and_store_family_artifacts.func()
        finally:
            GenerateReadmeTool.llm.invoke = old_invoke

        assert bool(result.get("ok")) is True
        assert mem.memory_retrieve("family_md")
        assert mem.memory_retrieve("family_index_js")
        model_list = mem.memory_retrieve("model_list") or []
        assert len(model_list) == 1
        github_url = mem.memory_retrieve("github_url") or []
        assert len(github_url) == len(model_list)

    with tempfile.TemporaryDirectory() as td:
        mem = GlobalMemory(str(Path(td) / "llama_mem.json"))
        GenerateReadmeTool.global_memory = mem
        mem.memory_store("generation_mode", "reference")
        mem.memory_store("model_list", ["Llama-3.2-3B-Instruct"])
        mem.memory_store("ref_md", "# Llama 3.1\n\nUse Llama-3.1 for deployment.")
        mem.memory_store("ref_index_js", "const name = 'Llama31ConfigGenerator';")
        old_invoke = GenerateReadmeTool.llm.invoke
        GenerateReadmeTool.llm.invoke = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("skip llm"))
        try:
            GenerateReadmeTool.memory_generate_and_store_family_artifacts.func()
        finally:
            GenerateReadmeTool.llm.invoke = old_invoke
        family_md = str(mem.memory_retrieve("family_md") or "")
        family_js = str(mem.memory_retrieve("family_index_js") or "")
        assert "Llama 3.2" in family_md or "Llama-3.2" in family_md
        assert "Llama 3.1" not in family_md
        assert "Llama31" not in family_js
        assert "Llama32" in family_js

    with tempfile.TemporaryDirectory() as td:
        mem = GlobalMemory(str(Path(td) / "cmd_mem.json"))
        GenerateReadmeTool.global_memory = mem
        mem.memory_store("generation_mode", "reference")
        mem.memory_store("model_list", ["Llama-3.2-3B-quantized.w8a8"])
        mem.memory_store("model_id_list", ["RedHatAI/Llama-3.2-3B-quantized.w8a8"])
        mem.memory_store("ref_md", "# Llama 3.2\n\n## Intro\nNo command examples here.")
        mem.memory_store("ref_index_js", "const cmd = 'python -m sglang.launch_server --tp 6';")
        old_invoke = GenerateReadmeTool.llm.invoke
        GenerateReadmeTool.llm.invoke = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("skip llm"))
        try:
            GenerateReadmeTool.memory_generate_and_store_family_artifacts.func()
        finally:
            GenerateReadmeTool.llm.invoke = old_invoke
        family_md = str(mem.memory_retrieve("family_md") or "")
        assert "## Quick Start Commands" in family_md
        assert "python -m sglang.launch_server" in family_md
        assert "```bash" in family_md

    print("PASS: readme autogen stores artifacts and compacts memory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
