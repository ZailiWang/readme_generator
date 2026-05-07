import json
from typing import Any, Dict, List

from crewai.tools import tool
from .common_utils import normalize_list


class PostRemoteRefineTool:
    global_memory = None

    @staticmethod
    def _collect_status_rows() -> List[Dict[str, Any]]:
        model_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("model_list") or []
        )
        model_id_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("model_id_list") or []
        )
        model_url_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("model_url_list") or []
        )
        github_url_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("github_url") or []
        )
        fail_reason_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("fail_reason_list") or []
        )
        execution_result = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("execution_result") or []
        )

        n = max(
            len(model_id_list),
            len(model_list),
            len(model_url_list),
            len(github_url_list),
            len(fail_reason_list),
            len(execution_result),
        )
        rows: List[Dict[str, Any]] = []
        for i in range(n):
            fail_reason = str(fail_reason_list[i]) if i < len(fail_reason_list) and fail_reason_list[i] else ""
            is_failed = bool(fail_reason.strip())
            rows.append(
                {
                    "idx": i,
                    "model_id": str(model_id_list[i]) if i < len(model_id_list) and model_id_list[i] else "",
                    "model_name": str(model_list[i]) if i < len(model_list) and model_list[i] else "",
                    "model_url": str(model_url_list[i]) if i < len(model_url_list) and model_url_list[i] else "",
                    "github_url": str(github_url_list[i]) if i < len(github_url_list) and github_url_list[i] else "",
                    "status": "failed" if is_failed else "passed",
                    "fail_reason": fail_reason,
                    "execution_result": execution_result[i] if i < len(execution_result) else "",
                }
            )
        return rows

    @staticmethod
    def _build_review_report(rows: List[Dict[str, Any]]) -> str:
        failed = [r for r in rows if r["status"] == "failed"]
        if not rows:
            return "# Remote Test Review\n\nNo remote execution rows were found."
        lines = [
            "# Remote Test Review",
            "",
            f"- total_models: {len(rows)}",
            f"- passed_models: {len([r for r in rows if r['status'] == 'passed'])}",
            f"- failed_models: {len(failed)}",
            "",
            "## Failed Model Details",
            "",
        ]
        if not failed:
            lines.append("No failed models.")
            return "\n".join(lines).strip()

        for row in failed:
            lines.extend(
                [
                    f"### {row.get('model_id') or row.get('model_name') or 'unknown-model'}",
                    f"- idx: {row.get('idx')}",
                    f"- model_name: {row.get('model_name')}",
                    f"- model_url: {row.get('model_url')}",
                    f"- github_url: {row.get('github_url')}",
                    "- fail_reason:",
                    "```text",
                    str(row.get("fail_reason") or ""),
                    "```",
                ]
            )
            execution_result = row.get("execution_result")
            if execution_result:
                lines.extend(
                    [
                        "- execution_result:",
                        "```text",
                        str(execution_result),
                        "```",
                    ]
                )
            lines.append("")
        return "\n".join(lines).strip()

    @tool("memory_retrieve_post_remote_context")
    def memory_retrieve_post_remote_context() -> Dict[str, Any]:
        """Retrieve model execution status and generated family artifacts for post-remote refinement."""
        rows = PostRemoteRefineTool._collect_status_rows()
        return {
            "rows": rows,
            "family_md": PostRemoteRefineTool.global_memory.memory_retrieve("family_md") or "",
            "family_index_js": PostRemoteRefineTool.global_memory.memory_retrieve("family_index_js") or "",
            "family_js_files": PostRemoteRefineTool.global_memory.memory_retrieve("family_js_files") or [],
            "source_js_files": PostRemoteRefineTool.global_memory.memory_retrieve("source_js_files") or [],
            "family_content": PostRemoteRefineTool.global_memory.memory_retrieve("family_content") or "",
        }

    @tool("memory_store_refined_family_artifacts")
    def memory_store_refined_family_artifacts(
        family_md: str,
        family_index_js: str,
        review_failure_report: str,
        family_js_files_json: str = "",
    ) -> Dict[str, Any]:
        """Store refined family artifacts and review report after removing failed models from docs/js."""
        rows = PostRemoteRefineTool._collect_status_rows()
        passed_indexes = [r["idx"] for r in rows if r["status"] == "passed"]

        old_model_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("model_list") or []
        )
        old_model_id_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("model_id_list") or []
        )
        old_model_url_list = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("model_url_list") or []
        )
        old_github_url = normalize_list(
            PostRemoteRefineTool.global_memory.memory_retrieve("github_url") or []
        )

        filtered_model_list = [old_model_list[i] for i in passed_indexes if i < len(old_model_list)]
        filtered_model_id_list = [old_model_id_list[i] for i in passed_indexes if i < len(old_model_id_list)]
        filtered_model_url_list = [old_model_url_list[i] for i in passed_indexes if i < len(old_model_url_list)]
        filtered_github_url = [old_github_url[i] for i in passed_indexes if i < len(old_github_url)]

        js_files = []
        if str(family_js_files_json or "").strip():
            raw = str(family_js_files_json).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                raw = raw.replace("json", "", 1).strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for i, item in enumerate(parsed):
                    if isinstance(item, dict):
                        js_files.append(
                            {
                                "path": str(item.get("path") or f"file_{i}.js"),
                                "content": str(item.get("content") or ""),
                            }
                        )
        if not js_files and (family_index_js or "").strip():
            js_files = [{"path": "index.js", "content": family_index_js or ""}]

        primary_index = next(
            (x for x in js_files if str(x.get("path", "")).split("/")[-1] == "index.js"),
            js_files[0] if js_files else {"content": family_index_js or ""},
        )
        primary_index_js = str(primary_index.get("content") or family_index_js or "")

        PostRemoteRefineTool.global_memory.memory_store("family_md", family_md or "")
        PostRemoteRefineTool.global_memory.memory_store("family_index_js", primary_index_js)
        PostRemoteRefineTool.global_memory.memory_store("family_js_files", js_files)
        combined = (
            f"{family_md or ''}\n\n---\n\n### index.js\n\n```javascript\n{primary_index_js}\n```".strip()
            if (family_md or "").strip() or primary_index_js.strip()
            else ""
        )
        PostRemoteRefineTool.global_memory.memory_store("family_content", combined)
        PostRemoteRefineTool.global_memory.memory_store(
            "review_failure_report",
            (review_failure_report or "").strip() or PostRemoteRefineTool._build_review_report(rows),
        )

        # Keep downstream memory aligned with docs that only include passed models.
        PostRemoteRefineTool.global_memory.memory_store("model_list", filtered_model_list)
        PostRemoteRefineTool.global_memory.memory_store("model_id_list", filtered_model_id_list)
        PostRemoteRefineTool.global_memory.memory_store("model_url_list", filtered_model_url_list)
        PostRemoteRefineTool.global_memory.memory_store("github_url", filtered_github_url)

        return {
            "ok": True,
            "passed_count": len(filtered_model_id_list),
            "failed_count": len(rows) - len(filtered_model_id_list),
        }
