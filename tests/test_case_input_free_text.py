#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.tools.input_parser_tool import InputParseTool  # noqa: E402
from readme_generator.main import WorkflowInput, prepare_workflow_input  # noqa: E402


def main() -> int:
    # URL-source free text
    text_url = (
        "请基于 https://github.com/sgl-project/sgl-cookbook/blob/main/docs/autoregressive/Qwen/Qwen3.md "
        "和 https://github.com/sgl-project/sgl-cookbook/blob/main/src/components/autoregressive/Qwen3ConfigGenerator/index.js "
        "处理 qwen3 7B 和 qwen3 8B，只走官方。"
    )
    parsed_url = InputParseTool.parse_input_text.func(text_url)
    assert parsed_url.get("generation_mode") == "web_sources"
    assert parsed_url.get("github_url") == []
    prepared_url = prepare_workflow_input(WorkflowInput(input_text=text_url, generation_mode="reference"))
    assert prepared_url.generation_mode == "web_sources"
    assert prepared_url.memory_profile == "url_source"

    # Legacy free text
    text_legacy = (
        "我要生成 Llama 3.1 8B instruct 和 Qwen3 8B 的readme。"
        "Qwen3 8B 走这个分支 https://github.com/jianan-gu/sglang/tree/cpu_optimized"
    )
    parsed_legacy = InputParseTool.parse_input_text.func(text_legacy)
    assert parsed_legacy.get("generation_mode") == "reference"
    assert isinstance(parsed_legacy.get("model_list"), list)
    assert len(parsed_legacy.get("model_list")) >= 1
    assert isinstance(parsed_legacy.get("github_url"), list)
    assert len(parsed_legacy.get("github_url")) == len(parsed_legacy.get("model_list"))
    prepared_legacy = prepare_workflow_input(WorkflowInput(input_text=text_legacy, generation_mode="reference"))
    assert prepared_legacy.generation_mode == "reference"
    assert prepared_legacy.memory_profile == "legacy"

    print("PASS: free-text parser supports unified flow branching")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
