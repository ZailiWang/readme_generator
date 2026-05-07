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
from readme_generator.tools.model_search_tool import ModelSearchTool  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        mem = GlobalMemory(str(Path(td) / "mem.json"))
        ModelSearchTool.global_memory = mem

        # fake HF backend
        def fake_search(_query: str, limit: int = 40):
            _ = limit
            return [
                {"modelId": "meta-llama/Llama-3.2-3B-Instruct"},
                {"modelId": "RedHatAI/Llama-3.2-3B-Instruct-FP8"},
                {"modelId": "hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4"},
            ]

        ModelSearchTool.hf_client._search = fake_search  # type: ignore[method-assign]

        names = [
            "Llama-3.2-3B-Instruct",
            "Llama-3.2-3B-Instruct-FP8",
            "Llama-3.2-3B-Instruct-AWQ",
        ]

        # legacy: strict 1:1
        mem.memory_store("generation_mode", "reference")
        legacy = ModelSearchTool.huggingface_model_batch_search.func(names)
        assert len(legacy["model_list"]) == len(names)
        assert len(legacy["model_id_list"]) == len(names)
        assert len(legacy["model_url_list"]) == len(names)

        # url_source: can expand
        mem.memory_store("generation_mode", "web_sources")
        expanded = ModelSearchTool.huggingface_model_batch_search.func(names)
        assert len(expanded["model_id_list"]) >= len(names)

    print("PASS: model_search branches by mode (legacy aligned, url_source expanded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

