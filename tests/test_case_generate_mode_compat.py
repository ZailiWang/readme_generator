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
    # legacy/reference: should require concrete model mention
    with tempfile.TemporaryDirectory() as td:
        m = GlobalMemory(str(Path(td) / "legacy.json"))
        GenerateReadmeTool.global_memory = m
        m.memory_store("generation_mode", "reference")
        m.memory_store("model_list", ["Qwen3-8B"])
        raised = False
        try:
            GenerateReadmeTool.memory_store_family_artifacts.func(
                "This doc only says Qwen family.",
                "export const backend='cpu';",
            )
        except ValueError:
            raised = True
        assert raised, "legacy mode should enforce concrete model alignment"

    # url_source/web_sources: family-level hint is acceptable
    with tempfile.TemporaryDirectory() as td:
        m = GlobalMemory(str(Path(td) / "url.json"))
        GenerateReadmeTool.global_memory = m
        m.memory_store("generation_mode", "web_sources")
        m.memory_store("model_list", ["Qwen3-8B"])
        out = GenerateReadmeTool.memory_store_family_artifacts.func(
            "This doc covers Qwen deployment on CUDA/AMD/Intel CPU.",
            "export const backend='cpu';",
        )
        assert bool(out.get("ok")) is True

    print("PASS: generate_readme compatibility for legacy and url_source modes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

