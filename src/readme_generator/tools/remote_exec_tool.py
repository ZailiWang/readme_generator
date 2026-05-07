import json
import re
from typing import Any, Dict, List, Optional

import requests
from crewai.tools import tool
from .common_utils import is_url_source_mode, normalize_list


class RemoteExecutionClient:
    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    @staticmethod
    def extract_commands_from_readme(readme: str) -> List[str]:
        if not readme:
            return []

        commands: List[str] = []
        for block in re.findall(r"```(?:bash|shell)\s*([\s\S]*?)```", readme, flags=re.IGNORECASE):
            for line in block.splitlines():
                cmd = line.strip()
                if not cmd or cmd.startswith("#"):
                    continue
                commands.append(cmd)
        return commands

    @staticmethod
    def _parse_stream_chunks(stream_chunks: List[str]) -> Dict[str, Any]:
        parsed_events: List[Any] = []
        text_events: List[str] = []
        reconstructed_text: List[str] = []
        for chunk in stream_chunks:
            if chunk in ("[DONE]", "DONE"):
                continue
            try:
                parsed = json.loads(chunk)
                parsed_events.append(parsed)
                if isinstance(parsed, dict):
                    choices = parsed.get("choices")
                    if isinstance(choices, list) and choices:
                        delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            reconstructed_text.append(content)
                    text_val = parsed.get("text") or parsed.get("content")
                    if isinstance(text_val, str) and text_val:
                        reconstructed_text.append(text_val)
                continue
            except Exception:
                pass
            text_events.append(chunk)

        merged_text = "".join(reconstructed_text).strip()
        if not merged_text and text_events:
            merged_text = "\n".join(text_events).strip()

        if parsed_events:
            return {"events": parsed_events, "text": merged_text}
        if text_events:
            return {"raw_response": merged_text}
        return {}

    @staticmethod
    def _read_sse_events(response: requests.Response) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        current_event = "message"
        data_lines: List[str] = []

        def flush_event() -> None:
            nonlocal current_event, data_lines
            if not data_lines:
                current_event = "message"
                return
            raw_data = "\n".join(data_lines).strip()
            try:
                parsed_data: Any = json.loads(raw_data)
            except Exception:
                parsed_data = raw_data
            events.append({"event": current_event, "data": parsed_data, "raw_data": raw_data})
            print(f"[remote_stream][{current_event}] {raw_data}")
            current_event = "message"
            data_lines = []

        for line in response.iter_lines(decode_unicode=True):
            if line is None:
                continue
            chunk = line.strip()
            if not chunk:
                flush_event()
                continue
            if chunk.startswith(":"):
                continue
            if chunk.startswith("event:"):
                current_event = chunk[6:].strip() or "message"
                continue
            if chunk.startswith("data:"):
                data_lines.append(chunk[5:].strip())
                continue
            data_lines.append(chunk)

        flush_event()
        return events

    @staticmethod
    def _parse_sse_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
        reconstructed: List[str] = []
        for item in events:
            data = item.get("data")
            if isinstance(data, dict):
                if isinstance(data.get("chunk"), str) and data.get("chunk"):
                    reconstructed.append(data["chunk"])
                    continue
                if isinstance(data.get("content"), str) and data.get("content"):
                    reconstructed.append(data["content"])
                    continue
                if isinstance(data.get("message"), str) and data.get("message"):
                    reconstructed.append(data["message"])
                    continue
            if isinstance(data, str) and data:
                reconstructed.append(data)
        return {"events": events, "text": "".join(reconstructed).strip()}

    def validate_payload(
        self,
        request_url: str,
        payload: Dict[str, Any],
        stream: bool = True,
    ) -> Dict[str, Any]:
        try:
            import pdb; pdb.set_trace()
            if stream:
                response = requests.post(
                    request_url,
                    json=payload,
                    timeout=self.timeout,
                    stream=True,
                    headers={"Accept": "text/event-stream"},
                )
                response.raise_for_status()
                content_type = str(response.headers.get("Content-Type", "")).lower()
                if "text/event-stream" in content_type:
                    sse_events = self._read_sse_events(response)
                    parsed_payload = self._parse_sse_events(sse_events)
                    stream_output = [json.dumps(evt, ensure_ascii=False) for evt in sse_events]
                else:
                    stream_chunks: List[str] = []
                    for line in response.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        chunk = line.strip()
                        if not chunk:
                            continue
                        stream_chunks.append(chunk)
                        print(f"[remote_stream] {chunk}")
                    parsed_payload = self._parse_stream_chunks(stream_chunks)
                    stream_output = stream_chunks

                return {
                    "ok": True,
                    "status_code": response.status_code,
                    "request_url": request_url,
                    "request_payload": payload,
                    "response": parsed_payload,
                    "stream_output": stream_output,
                    "used_stream": True,
                }

            response = requests.post(
                request_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            try:
                response_payload = response.json()
            except ValueError:
                response_payload = {"raw_response": response.text}
            return {
                "ok": True,
                "status_code": response.status_code,
                "request_url": request_url,
                "request_payload": payload,
                "response": response_payload,
                "used_stream": False,
            }
        except Exception as e:
            return {
                "ok": False,
                "status_code": None,
                "request_url": request_url,
                "response": {},
                "error": str(e),
                "used_stream": stream,
            }

    def validate_model_readme(
        self,
        request_url: str,
        model_id: str,
        content: str,
        extra_payload: Optional[Dict[str, Any]] = None,
        stream: bool = True,
        include_extracted_commands: bool = False,
    ) -> Dict[str, Any]:
        payload = {
            "model_id": model_id,
            "content": content,
        }
        if include_extracted_commands:
            payload["commands"] = self.extract_commands_from_readme(content)
        if extra_payload:
            payload.update(extra_payload)
        return self.validate_payload(request_url=request_url, payload=payload, stream=stream)


class RemoteExecutionTool:
    client = RemoteExecutionClient()
    global_memory = None

    @staticmethod
    def _compose_model_content_from_family(
        model_id: str,
        model_name: str,
        model_url: str,
        github_url: str,
        family_md: str,
        family_index_js: str,
        family_js_files: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        branch_mode = "dev_branch" if str(github_url or "").strip() else "official"
        model_header = (model_id or model_name or "").strip()
        model_url = (model_url or "").strip()
        github_url = (github_url or "").strip()
        family_md = (family_md or "").strip()
        family_index_js = (family_index_js or "").strip()

        if not family_md:
            return ""
        js_sections: List[str] = []
        js_files = family_js_files or []
        if js_files:
            for item in js_files:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "index.js")
                content = str(item.get("content") or "")
                js_sections.append(f"### {path}\n```javascript\n{content}\n```")
        elif family_index_js:
            js_sections.append(f"### index.js\n```javascript\n{family_index_js}\n```")
        if not js_sections:
            return ""

        return (
            f"## Target Model\n"
            f"- model_id: {model_id}\n"
            f"- model_name: {model_name}\n"
            f"- model_url: {model_url}\n"
            f"- mode: {branch_mode}\n"
            f"- github_url: {github_url}\n\n"
            "## Rewriting Requirement\n"
            f"Use the full family README.md and index.js below to derive commands and test logic for ONLY this target model ({model_header}). "
            "Do not execute other model variants.\n\n"
            "## Family README.md (full)\n\n"
            f"{family_md}\n\n"
            "## Family index.js (full)\n\n"
            f"{chr(10).join(js_sections)}"
        ).strip()

    @staticmethod
    def _resolve_model_content_list() -> Dict[str, Any]:
        model_list = normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("model_list") or []
            ,
            fallback_single_str=True,
            stringify_items=True,
        )
        model_id_list = normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("model_id_list") or []
            ,
            fallback_single_str=True,
            stringify_items=True,
        )
        model_url_list = normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("model_url_list") or []
            ,
            fallback_single_str=True,
            stringify_items=True,
        )
        github_url_list = normalize_list(
            RemoteExecutionTool.global_memory.memory_retrieve("github_url") or []
            ,
            fallback_single_str=True,
            stringify_items=True,
        )
        family_md = str(
            RemoteExecutionTool.global_memory.memory_retrieve("family_md") or ""
        ).strip()
        family_index_js = str(RemoteExecutionTool.global_memory.memory_retrieve("family_index_js") or "").strip()
        family_js_files_raw = RemoteExecutionTool.global_memory.memory_retrieve("family_js_files") or []
        family_js_files = family_js_files_raw if isinstance(family_js_files_raw, list) else []
        if not family_md:
            raise ValueError("family_md is required for remote execution.")
        if not family_index_js and not family_js_files:
            raise ValueError("family_index_js/family_js_files is required for remote execution.")
        if not model_id_list:
            raise ValueError("model_id_list is required for remote execution.")

        derived: List[str] = []
        for i in range(len(model_id_list)):
            derived.append(
                RemoteExecutionTool._compose_model_content_from_family(
                    model_id=model_id_list[i] if i < len(model_id_list) else "",
                    model_name=model_list[i] if i < len(model_list) else "",
                    model_url=model_url_list[i] if i < len(model_url_list) else "",
                    github_url=github_url_list[i] if i < len(github_url_list) else "",
                    family_md=family_md,
                    family_index_js=family_index_js,
                    family_js_files=family_js_files,
                )
            )
        return {
            "model_content_list": derived,
            "warning": "",
        }

    @staticmethod
    def _resolve_request_url() -> str:
        ssh_config = RemoteExecutionTool.global_memory.memory_retrieve("ssh_config") or {}
        # Preferred override: full URL in ssh_config.request_url
        # Otherwise composed by ssh_config.hostname/request_scheme/request_port/request_endpoint
        if ssh_config.get("request_url"):
            return ssh_config["request_url"]

        host = ssh_config.get("hostname")
        if not host:
            raise ValueError("Missing request_url or hostname in ssh_config.")

        scheme = ssh_config.get("request_scheme", "http")
        port = ssh_config.get("request_port", 8000)
        request_stream = bool(ssh_config.get("request_stream", False))
        mode = RemoteExecutionTool._normalized_remote_mode()
        base_endpoint = "/url_source_test" if mode == "url_source" else "/legacy_test"
        default_endpoint = f"{base_endpoint}/stream" if request_stream else base_endpoint
        endpoint = ssh_config.get("request_endpoint", default_endpoint)
        return f"{scheme}://{host}:{port}{endpoint}"

    @staticmethod
    def _build_execution_context() -> Dict[str, Any]:
        mode = RemoteExecutionTool._normalized_remote_mode()
        if mode == "url_source":
            resolved_content = {"model_content_list": [], "warning": ""}
        else:
            resolved_content = RemoteExecutionTool._resolve_model_content_list()
        remote_payload = RemoteExecutionTool.global_memory.memory_retrieve("remote_payload") or {}
        if not isinstance(remote_payload, dict):
            remote_payload = {}
        if mode == "url_source":
            memory_model_list = normalize_list(
                RemoteExecutionTool.global_memory.memory_retrieve("model_list") or [],
                fallback_single_str=True,
                stringify_items=True,
            )
            model_list = normalize_list(
                memory_model_list or remote_payload.get("model_list") or [],
                fallback_single_str=True,
                stringify_items=True,
            )
            remote_payload = {
                "generation_mode": "url_source",
                "model_list": model_list,
                "source_urls": remote_payload.get("source_urls") or {
                    "md": RemoteExecutionTool.global_memory.memory_retrieve("source_md_url") or "",
                    "js": RemoteExecutionTool.global_memory.memory_retrieve("source_js_url") or "",
                },
                "metadata": remote_payload.get("metadata") or {"official_only": True},
            }
        return {
            "generation_mode": mode,
            "model_list": RemoteExecutionTool.global_memory.memory_retrieve("model_list") or [],
            "model_id_list": RemoteExecutionTool.global_memory.memory_retrieve("model_id_list") or [],
            "model_url_list": RemoteExecutionTool.global_memory.memory_retrieve("model_url_list") or [],
            "github_url": RemoteExecutionTool.global_memory.memory_retrieve("github_url") or [],
            "family_md": RemoteExecutionTool.global_memory.memory_retrieve("family_md") or "",
            "family_index_js": RemoteExecutionTool.global_memory.memory_retrieve("family_index_js") or "",
            "family_js_files": RemoteExecutionTool.global_memory.memory_retrieve("family_js_files") or [],
            "family_content": RemoteExecutionTool.global_memory.memory_retrieve("family_content") or "",
            "model_content_list": resolved_content["model_content_list"],
            "content_resolution_warning": resolved_content["warning"],
            "execution_result": RemoteExecutionTool.global_memory.memory_retrieve("execution_result") or [],
            "executed_command": RemoteExecutionTool.global_memory.memory_retrieve("executed_command") or [],
            "fail_reason_list": RemoteExecutionTool.global_memory.memory_retrieve("fail_reason_list") or [],
            "ssh_config": RemoteExecutionTool.global_memory.memory_retrieve("ssh_config") or {},
            "remote_payload": remote_payload,
        }

    @staticmethod
    def _resolve_legacy_content(remote_payload: Dict[str, Any]) -> Dict[str, str]:
        family_md = str(RemoteExecutionTool.global_memory.memory_retrieve("family_md") or "")
        family_index_js = str(RemoteExecutionTool.global_memory.memory_retrieve("family_index_js") or "")
        if family_md.strip() and family_index_js.strip():
            return {
                "family_md": family_md,
                "family_index_js": family_index_js,
                "input_text": str(RemoteExecutionTool.global_memory.memory_retrieve("input_text") or ""),
            }
        content = remote_payload.get("content")
        if isinstance(content, dict):
            ref_md = str(content.get("family_md") or content.get("ref_md") or "")
            ref_index_js = str(content.get("family_index_js") or content.get("ref_index_js") or "")
            input_text = str(content.get("input_text") or "")
            if ref_md and ref_index_js:
                return {
                    "family_md": ref_md,
                    "family_index_js": ref_index_js,
                    "input_text": input_text,
                }
        return {
            "family_md": str(RemoteExecutionTool.global_memory.memory_retrieve("family_md") or "")
            or str(RemoteExecutionTool.global_memory.memory_retrieve("ref_md") or ""),
            "family_index_js": str(RemoteExecutionTool.global_memory.memory_retrieve("family_index_js") or "")
            or str(RemoteExecutionTool.global_memory.memory_retrieve("ref_index_js") or ""),
            "input_text": str(RemoteExecutionTool.global_memory.memory_retrieve("input_text") or ""),
        }

    @tool("memory_retrieve_execution_context")
    def memory_retrieve_execution_context():
        """Retrieve all information needed for remote execution from GLOBAL_MEMORY.
        Returns: dictionary containing model_id_list, per-model model_content_list and request config."""
        return RemoteExecutionTool._build_execution_context()

    @tool("memory_preview_remote_content")
    def memory_preview_remote_content(preview_chars: int = 1000) -> Dict[str, Any]:
        """Preview per-model content that will be sent to remote API before execution.
        Returns model-level summaries including content length and truncated preview."""
        context = RemoteExecutionTool._build_execution_context()
        if context.get("generation_mode") == "url_source":
            payload = context.get("remote_payload") or {}
            return {
                "count": len(payload.get("model_list") or []),
                "content_resolution_warning": "",
                "items": [],
                "payload_preview": payload,
            }
        model_ids = context.get("model_id_list") or []
        model_contents = context.get("model_content_list") or []
        preview_items: List[Dict[str, Any]] = []

        for idx, model_id in enumerate(model_ids):
            content = model_contents[idx] if idx < len(model_contents) else ""
            preview_items.append(
                {
                    "idx": idx,
                    "model_id": model_id,
                    "content_length": len(content or ""),
                    "content_preview": (content or "")[: max(0, int(preview_chars))],
                }
            )

        return {
            "count": len(preview_items),
            "content_resolution_warning": context.get("content_resolution_warning", ""),
            "items": preview_items,
        }

    @tool("execute_remote_readme_validation")
    def execute_remote_readme_validation(model_id: str, model_content: str) -> Dict[str, Any]:
        """Validate single-model content (md+js) on remote local-bkc agent by one HTTP request.
        Inputs: model_id, model_content
        Returns: remote validation result payload."""
        request_url = RemoteExecutionTool._resolve_request_url()
        ssh_config = RemoteExecutionTool.global_memory.memory_retrieve("ssh_config") or {}
        extra_payload = ssh_config.get("request_payload", {})
        request_stream = bool(ssh_config.get("request_stream", False))
        include_extracted_commands = bool(ssh_config.get("include_extracted_commands", False))
        mode = RemoteExecutionTool._normalized_remote_mode()
        remote_payload = RemoteExecutionTool.global_memory.memory_retrieve("remote_payload") or {}
        if not isinstance(remote_payload, dict):
            remote_payload = {}
        if mode == "url_source":
            memory_model_list = normalize_list(
                RemoteExecutionTool.global_memory.memory_retrieve("model_list") or [],
                fallback_single_str=True,
                stringify_items=True,
            )
            payload = {
                "generation_mode": "url_source",
                "model_list": normalize_list(
                    memory_model_list or remote_payload.get("model_list") or [],
                    fallback_single_str=True,
                    stringify_items=True,
                ),
                "source_urls": remote_payload.get("source_urls") or {
                    "md": RemoteExecutionTool.global_memory.memory_retrieve("source_md_url") or "",
                    "js": RemoteExecutionTool.global_memory.memory_retrieve("source_js_url") or "",
                },
                "metadata": remote_payload.get("metadata") or {"official_only": True},
            }
            if isinstance(extra_payload, dict) and extra_payload:
                payload.update(extra_payload)
            return RemoteExecutionTool.client.validate_payload(
                request_url=request_url,
                payload=payload,
                stream=request_stream,
            )
        legacy_payload = {
            "generation_mode": "legacy",
            "content": RemoteExecutionTool._resolve_legacy_content(remote_payload),
            "model_list": (RemoteExecutionTool.global_memory.memory_retrieve("model_list") or []) or remote_payload.get("model_list") or [],
            "github_url": remote_payload.get("github_url") or (RemoteExecutionTool.global_memory.memory_retrieve("github_url") or []),
            "metadata": remote_payload.get("metadata") or {"official": [], "dev": []},
            # keep per-model fields for current remote loop compatibility
            "model_id": model_id,
            "content_single_model": model_content,
        }
        if include_extracted_commands:
            legacy_payload["commands"] = RemoteExecutionTool.client.extract_commands_from_readme(model_content)
        if isinstance(extra_payload, dict) and extra_payload:
            legacy_payload.update(extra_payload)
        return RemoteExecutionTool.client.validate_payload(
            request_url=request_url,
            payload=legacy_payload,
            stream=request_stream,
        )

    @tool("memory_store_execution_result")
    def memory_store_execution_result(
        idx: int,
        command_str: str,
        result: Any,
        fail_reason: Optional[str],
        updated_readme: Optional[str],
    ):
        """Store remote execution results into GLOBAL_MEMORY at specified index.
        Inputs: index, executed command, execution result, fail reason, updated readme
        Returns: success message."""

        def update_list(key, value):
            lst = RemoteExecutionTool.global_memory.memory_retrieve(key=key) or []
            while len(lst) <= idx:
                lst.append(None)
            lst[idx] = value
            RemoteExecutionTool.global_memory.memory_store(key=key, value=lst)

        update_list("executed_command", command_str)
        stored_result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        update_list("execution_result", stored_result)
        update_list("fail_reason_list", fail_reason)
        return True
    @staticmethod
    def _normalized_remote_mode() -> str:
        remote_payload = RemoteExecutionTool.global_memory.memory_retrieve("remote_payload") or {}
        mode = ""
        if isinstance(remote_payload, dict):
            mode = str(remote_payload.get("generation_mode") or "").strip().lower()
        if not mode:
            mode = str(RemoteExecutionTool.global_memory.memory_retrieve("generation_mode") or "").strip().lower()
        if is_url_source_mode(mode):
            return "url_source"
        return "legacy"
