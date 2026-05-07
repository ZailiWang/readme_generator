#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.crew import ReadmeWorkflowCrew, WorkflowInput  # noqa: E402
from readme_generator.main import (  # noqa: E402
    LEGACY_USER_INPUT_TEXT_EXAMPLE,
    QWEN3_URL_SOURCE_INPUT_TEXT_EXAMPLE,
)
from readme_generator.tools.memory_tool import GlobalMemory  # noqa: E402


def _run_case(case_name: str, workflow_input: WorkflowInput, outdir: Path) -> dict:
    case_dir = outdir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ.setdefault("http_proxy", "http://proxy-dmz.intel.com:912")
    os.environ.setdefault("https_proxy", "http://proxy-dmz.intel.com:912")
    os.environ["NO_PROXY"] = "10.54.34.78,10.165.58.104,127.0.0.1,localhost"
    os.environ["no_proxy"] = os.environ["NO_PROXY"]

    mem_path = case_dir / "run_memory.json"
    mem_path.write_text("{}", encoding="utf-8")
    memory = GlobalMemory(persist_path=str(mem_path))
    memory.memory_store("input_text", workflow_input.input_text or "")
    results = []
    error = ""
    try:
        runner = ReadmeWorkflowCrew(
            workflow_input=workflow_input,
            enabled_stages=["input_parser", "model_search", "readme_generation"],
            memory=memory,
        )
        results = runner.run()
    except Exception as e:
        error = str(e)

    family_md = memory.memory_retrieve("family_md") or ""
    family_index_js = memory.memory_retrieve("family_index_js") or ""
    source_md_files = memory.memory_retrieve("source_md_files") or []
    source_js_files = memory.memory_retrieve("source_js_files") or []
    model_list = memory.memory_retrieve("model_list") or []
    model_id_list = memory.memory_retrieve("model_id_list") or []

    (case_dir / "family_md.md").write_text(family_md, encoding="utf-8")
    (case_dir / "family_index.js").write_text(family_index_js, encoding="utf-8")
    (case_dir / "stage_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "source_files.json").write_text(
        json.dumps(
            {
                "source_md_files_count": len(source_md_files) if isinstance(source_md_files, list) else 0,
                "source_js_files_count": len(source_js_files) if isinstance(source_js_files, list) else 0,
                "source_md_files": source_md_files,
                "source_js_files": source_js_files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (case_dir / "models.json").write_text(
        json.dumps({"model_list": model_list, "model_id_list": model_id_list}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    remote_payload = memory.memory_retrieve("remote_payload") or {}
    (case_dir / "remote_payload.json").write_text(
        json.dumps(remote_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "case_name": case_name,
        "ok": not bool(error),
        "error": error,
        "output_dir": str(case_dir),
        "went_through_input_parser": any((x.get("stage") == "input_parser") for x in (results or [])),
        "generation_mode": memory.memory_retrieve("generation_mode"),
        "source_md_url": memory.memory_retrieve("source_md_url"),
        "source_js_url": memory.memory_retrieve("source_js_url"),
        "source_md_files_count": len(source_md_files) if isinstance(source_md_files, list) else 0,
        "source_js_files_count": len(source_js_files) if isinstance(source_js_files, list) else 0,
        "model_list_count": len(model_list) if isinstance(model_list, list) else 0,
        "model_id_list_count": len(model_id_list) if isinstance(model_id_list, list) else 0,
        "family_md_len": len(family_md),
        "family_index_js_len": len(family_index_js),
        "contains_intel_cpu_keyword": (
            ("intel" in family_md.lower())
            or ("intel cpu" in family_md.lower())
            or ("--device cpu" in family_index_js.lower())
        ),
        "remote_payload_mode": remote_payload.get("generation_mode") if isinstance(remote_payload, dict) else "",
    }
    (case_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    run_mode = (os.getenv("README_FLOW_CASE") or "url_source").strip().lower()
    outdir = ROOT / "review_outputs" / f"unified_flow_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    cases = []
    if run_mode in {"legacy", "both"}:
        cases.append(
            (
                "legacy",
                WorkflowInput(
                    memory_profile="legacy_test",
                    generation_mode="reference",
                    input_text=LEGACY_USER_INPUT_TEXT_EXAMPLE,
                    model_list=[],
                    github_url=[],
                    skip_stages=[],
                    remote_folder="",
                    ssh_config={},
                    github_config={},
                ),
            )
        )
    if run_mode in {"url_source", "both"}:
        cases.append(
            (
                "url_source",
                WorkflowInput(
                    memory_profile="url_source_test",
                    generation_mode="web_sources",
                    input_text=QWEN3_URL_SOURCE_INPUT_TEXT_EXAMPLE,
                    model_list=[],
                    github_url=[],
                    skip_stages=[],
                    remote_folder="",
                    ssh_config={},
                    github_config={},
                ),
            )
        )
    if not cases:
        raise ValueError("README_FLOW_CASE must be one of: legacy, url_source, both")

    summaries = []
    for case_name, wf in cases:
        summaries.append(_run_case(case_name, wf, outdir))

    overall_ok = all(bool(s.get("ok")) for s in summaries)
    merged = {"ok": overall_ok, "mode": run_mode, "output_dir": str(outdir), "cases": summaries}
    (outdir / "run_summary.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False))
    print(f"[OUTDIR] {outdir}")
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
