#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests

REPO_ROOT = Path("/home/changrui/readme_generator")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from readme_generator.crew import ReadmeWorkflowCrew, WorkflowInput  # noqa: E402
from readme_generator.tools.memory_tool import GlobalMemory  # noqa: E402


QWEN3_MD_GITHUB_URL = "https://github.com/sgl-project/sgl-cookbook/blob/main/docs/autoregressive/Qwen/Qwen3.md"
QWEN3_JS_GITHUB_URL = "https://github.com/sgl-project/sgl-cookbook/blob/main/src/components/autoregressive/Qwen3ConfigGenerator/"
QWEN3_MD_RAW_URL = "https://raw.githubusercontent.com/sgl-project/sgl-cookbook/main/docs/autoregressive/Qwen/Qwen3.md"
QWEN3_JS_RAW_URL = "https://raw.githubusercontent.com/sgl-project/sgl-cookbook/main/src/components/autoregressive/Qwen3ConfigGenerator/index.js"


def _setup_network_env() -> None:
    # Keep proxy for public internet, bypass proxy for internal model services.
    os.environ.setdefault("http_proxy", "http://proxy-dmz.intel.com:912")
    os.environ.setdefault("https_proxy", "http://proxy-dmz.intel.com:912")
    no_proxy_hosts = {"10.54.34.78", "10.165.58.104", "127.0.0.1", "localhost"}
    for key in ("NO_PROXY", "no_proxy"):
        current = os.environ.get(key, "")
        merged = {x.strip() for x in current.split(",") if x.strip()} | no_proxy_hosts
        os.environ[key] = ",".join(sorted(merged))
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def _download_text(url: str, timeout: int = 120) -> str:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def _prepare_output_dir() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    outdir = REPO_ROOT / "review_outputs" / f"qwen3_intel_pre_remote_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def _save_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    _setup_network_env()
    outdir = _prepare_output_dir()

    try:
        ref_md = _download_text(QWEN3_MD_RAW_URL)
        ref_index_js = _download_text(QWEN3_JS_RAW_URL)
    except Exception as e:
        _save_json(
            outdir / "run_summary.json",
            {
                "ok": False,
                "error": f"download_failed: {e}",
                "md_url": QWEN3_MD_GITHUB_URL,
                "js_url": QWEN3_JS_GITHUB_URL,
            },
        )
        print(f"[FAIL] download failed: {e}")
        print(f"[OUTDIR] {outdir}")
        return 1

    mem_path = outdir / "run_memory.json"
    mem_path.write_text("{}", encoding="utf-8")
    memory = GlobalMemory(persist_path=str(mem_path))
    memory.memory_store("source_md_url", QWEN3_MD_GITHUB_URL)
    memory.memory_store("source_js_url", QWEN3_JS_GITHUB_URL)
    memory.memory_store("source_md_files", [{"path": "docs/autoregressive/Qwen/Qwen3.md", "content": ref_md}])
    memory.memory_store(
        "source_js_files",
        [{"path": "src/components/autoregressive/Qwen3ConfigGenerator/index.js", "content": ref_index_js}],
    )

    workflow_input = WorkflowInput(
        generation_mode="reference",
        ref_md=ref_md,
        ref_index_js=ref_index_js,
        input_text="",
        model_list=[],
        github_url=[],
        remote_folder="",
        ssh_config={},
        github_config={},
    )

    results: List[Dict] = []
    err = ""
    try:
        runner = ReadmeWorkflowCrew(
            workflow_input=workflow_input,
            enabled_stages=["model_search", "readme_generation"],
            memory=memory,
        )
        results = runner.run()
    except Exception as e:
        err = str(e)

    family_md = memory.memory_retrieve("family_md") or ""
    family_index_js = memory.memory_retrieve("family_index_js") or ""
    family_js_files = memory.memory_retrieve("family_js_files") or []
    model_list = memory.memory_retrieve("model_list") or []
    model_id_list = memory.memory_retrieve("model_id_list") or []

    (outdir / "family_md.md").write_text(family_md, encoding="utf-8")
    (outdir / "family_index.js").write_text(family_index_js, encoding="utf-8")
    _save_json(outdir / "family_js_files.json", {"family_js_files": family_js_files})
    _save_json(outdir / "stage_results.json", {"results": results})
    _save_json(outdir / "inferred_models.json", {"model_list": model_list, "model_id_list": model_id_list})

    summary = {
        "ok": not bool(err),
        "error": err,
        "md_url": QWEN3_MD_GITHUB_URL,
        "js_url": QWEN3_JS_GITHUB_URL,
        "family_md_len": len(family_md),
        "family_index_js_len": len(family_index_js),
        "family_js_files_count": len(family_js_files) if isinstance(family_js_files, list) else 0,
        "model_list_count": len(model_list) if isinstance(model_list, list) else 0,
        "model_id_list_count": len(model_id_list) if isinstance(model_id_list, list) else 0,
        "output_dir": str(outdir),
    }
    _save_json(outdir / "run_summary.json", summary)

    print(json.dumps(summary, ensure_ascii=False))
    print(f"[OUTDIR] {outdir}")
    return 0 if not err else 2


if __name__ == "__main__":
    raise SystemExit(main())

