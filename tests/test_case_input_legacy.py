#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.main import WorkflowInput, prepare_enabled_stages, prepare_workflow_input  # noqa: E402
from readme_generator.tools.input_parser_tool import InternelParserLLM  # noqa: E402


def main() -> int:
    wf = WorkflowInput(
        generation_mode="reference",
        input_text="",
        model_list=["Llama-3.1-8B-Instruct", "Llama-3.1-8B-Instruct-FP8"],
        github_url=["", "https://github.com/jianan-gu/sglang/tree/cpu_optimized"],
    )
    wf = prepare_workflow_input(wf)
    payload = json.loads(wf.input_text)
    parsed = InternelParserLLM._extract_from_workflow_payload(wf.input_text)
    stages = prepare_enabled_stages(wf, ["model_search", "readme_generation"])

    assert payload["model_list"] == wf.model_list
    assert payload["github_url"] == wf.github_url
    assert parsed.get("model_list") == wf.model_list
    assert parsed.get("github_url") == wf.github_url
    assert stages and stages[0] == "input_parser"
    print("PASS: legacy input parsing and stage alignment")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
