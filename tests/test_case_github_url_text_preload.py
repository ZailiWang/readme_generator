#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import readme_generator.crew as workflow_crew  # noqa: E402
from readme_generator.crew import ReadmeWorkflowCrew, WorkflowInput  # noqa: E402
from readme_generator.main import QWEN3_URL_SOURCE_INPUT_TEXT_EXAMPLE  # noqa: E402
from readme_generator.tools.memory_tool import GlobalMemory  # noqa: E402


def main() -> int:
    original_loader = workflow_crew.load_reference_files_from_github_folders
    called = {"value": False}

    def _fake_loader(md_folder_url: str, js_folder_url: str, github_token: str = ""):
        called["value"] = True
        assert "Qwen3.md" in md_folder_url
        assert "Qwen3ConfigGenerator" in js_folder_url
        return {
            "ref_md": "# Qwen3 from URL\nQwen/Qwen3-8B",
            "ref_index_js": "export const model='Qwen/Qwen3-8B';",
            "source_md_files": [{"path": "Qwen3.md", "content": "Qwen/Qwen3-8B"}],
            "source_js_files": [{"path": "index.js", "content": "Qwen/Qwen3-8B"}],
        }

    workflow_crew.load_reference_files_from_github_folders = _fake_loader
    try:
        with tempfile.TemporaryDirectory() as td:
            memory = GlobalMemory(persist_path=str(Path(td) / "memory.json"))
            wf = WorkflowInput(
                generation_mode="reference",
                input_text=QWEN3_URL_SOURCE_INPUT_TEXT_EXAMPLE,
                model_list=[],
                github_url=[],
            )
            ReadmeWorkflowCrew(
                workflow_input=wf,
                enabled_stages=["input_parser"],
                memory=memory,
            )

            assert called["value"] is True
            assert memory.memory_retrieve("generation_mode") == "web_sources"
            assert "Qwen3.md" in str(memory.memory_retrieve("source_md_url") or "")
            assert "Qwen3ConfigGenerator" in str(memory.memory_retrieve("source_js_url") or "")
            assert "Qwen3 from URL" in str(memory.memory_retrieve("ref_md") or "")
            assert "Qwen/Qwen3-8B" in str(memory.memory_retrieve("ref_index_js") or "")
    finally:
        workflow_crew.load_reference_files_from_github_folders = original_loader

    print("PASS: github URL text input preloads ref_md/ref_index_js from URL sources")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
