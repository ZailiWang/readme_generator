#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.main import WorkflowInput, prepare_enabled_stages, prepare_workflow_input  # noqa: E402
from readme_generator.tools.input_parser_tool import InputParseTool, InternelParserLLM  # noqa: E402
from readme_generator.tools.memory_tool import GlobalMemory  # noqa: E402
from readme_generator.tools.model_search_tool import ModelSearchTool  # noqa: E402


def main() -> int:
    md_url = "https://github.com/sgl-project/sgl-cookbook/blob/main/docs/autoregressive/Qwen/Qwen3.md"
    js_url = "https://github.com/sgl-project/sgl-cookbook/blob/main/src/components/autoregressive/Qwen3ConfigGenerator/"
    wf = WorkflowInput(
        generation_mode="web_sources",
        source_md_url=md_url,
        source_js_url=js_url,
        input_text="",
        model_list=[],
        github_url=[],
    )
    wf = prepare_workflow_input(wf)
    payload = json.loads(wf.input_text)
    parsed = InternelParserLLM._extract_from_workflow_payload(wf.input_text)
    stages = prepare_enabled_stages(wf, ["model_search", "readme_generation"])

    assert payload["generation_mode"] == "web_sources"
    assert payload["source_md_url"] == md_url
    assert payload["source_js_url"] == js_url
    assert payload["github_url"] == []

    assert parsed.get("generation_mode") == "web_sources"
    assert parsed.get("source_md_url") == md_url
    assert parsed.get("source_js_url") == js_url
    assert parsed.get("github_url") == []
    assert parsed.get("model_list") == []
    assert stages and stages[0] == "input_parser"

    with tempfile.TemporaryDirectory() as td:
        memory = GlobalMemory(persist_path=str(Path(td) / "memory.json"))
        memory.memory_store("generation_mode", "web_sources")
        memory.memory_store("input_text", "请处理 Qwen3 系列模型，并补充 Intel CPU。")
        memory.memory_store(
            "source_md_files",
            [
                {"path": "Qwen3.md", "content": "推荐模型：Qwen/Qwen3-4B 与 Qwen/Qwen3-8B。"},
            ],
        )
        memory.memory_store("source_md_url", md_url)
        memory.memory_store("source_js_url", js_url)
        memory.memory_store("ref_md", "Legacy fallback includes meta-llama/Llama-3.1-8B-Instruct only.")
        memory.memory_store(
            "source_js_files",
            [
                {"path": "index.js", "content": "const modelIds = ['Qwen/Qwen3-4B', 'Qwen/Qwen3-8B', 'Qwen3 7B'];"},
            ],
        )
        InputParseTool.global_memory = memory
        ModelSearchTool.global_memory = memory

        InputParseTool.store_memory.func(json.dumps(parsed, ensure_ascii=False))
        parser_models = memory.memory_retrieve("model_list") or []
        assert any("qwen3-4b" in m.lower() for m in parser_models)
        assert any("qwen3-8b" in m.lower() for m in parser_models)
        assert any(("qwen3" in m.lower() and "7b" in m.lower()) for m in parser_models)
        assert all("llama" not in m.lower() for m in parser_models)

        search_models = ModelSearchTool.memory_retrieve_model_list.func()
        assert any("qwen3-4b" in m.lower() for m in search_models)
        assert any("qwen3-8b" in m.lower() for m in search_models)
        assert any(("qwen3" in m.lower() and "7b" in m.lower()) for m in search_models)
        assert all("llama" not in m.lower() for m in search_models)
    print("PASS: URL-source input parsing (official-only github_url=[]) and stage alignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
