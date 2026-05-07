from crewai.tools import tool
import json
import re
from typing import Any, Dict, List
from .chatopenai import LLM_Callable
from .common_utils import infer_family_hint_from_corpus, is_url_source_mode, normalize_list


class GenerateReadmeTool:
    global_memory = None
    llm = LLM_Callable(
        base_url="http://10.54.34.78:30000/v1",
        api_key="empty",
        model_name="local-model",
    )

    @staticmethod
    def _normalize_js_files(js_files: list) -> list:
        normalized = []
        for i, item in enumerate(js_files or []):
            if isinstance(item, dict):
                path = str(item.get("path") or f"file_{i}.js")
                content = str(item.get("content") or "")
            else:
                path = f"file_{i}.js"
                content = str(item or "")
            if not path.endswith(".js"):
                path = f"{path}.js"
            normalized.append({"path": path, "content": content})
        return normalized

    @staticmethod
    def _compose_family_content(family_md: str, family_index_js: str, family_js_files: list | None = None) -> str:
        md_text = (family_md or "").strip()
        js_text = (family_index_js or "").strip()
        js_files = GenerateReadmeTool._normalize_js_files(family_js_files or [])
        if js_files:
            sections = []
            for item in js_files:
                sections.append(
                    f"### {item['path']}\n\n```javascript\n{(item.get('content') or '').strip()}\n```"
                )
            js_block = "\n\n".join(sections).strip()
        else:
            js_block = f"### index.js\n\n```javascript\n{js_text}\n```" if js_text else ""

        if not md_text and not js_block:
            return ""
        if not js_block:
            return md_text
        if not md_text:
            return js_block
        return f"{md_text}\n\n---\n\n{js_block}"

    @staticmethod
    def _validate_target_models(family_md: str, family_index_js: str) -> None:
        mode = str(GenerateReadmeTool.global_memory.memory_retrieve("generation_mode") or "").strip().lower()
        model_list = GenerateReadmeTool.global_memory.memory_retrieve("model_list") or []
        model_id_list = GenerateReadmeTool.global_memory.memory_retrieve("model_id_list") or []
        all_text = f"{family_md or ''}\n{family_index_js or ''}"
        lowered = all_text.lower()

        candidates = []
        for raw in list(model_list) + list(model_id_list):
            text = str(raw or "").strip()
            if not text:
                continue
            candidates.append(text)
            if "/" in text:
                candidates.append(text.split("/", 1)[1].strip())

        deduped = []
        seen = set()
        for c in candidates:
            lc = c.lower()
            if lc in seen:
                continue
            seen.add(lc)
            deduped.append(c)

        if not deduped:
            return
        if not any(c.lower() in lowered for c in deduped):
            if is_url_source_mode(mode):
                family_hint = infer_family_hint_from_corpus(deduped) or infer_family_hint_from_corpus([all_text])
                if family_hint and family_hint in lowered:
                    return
            raise ValueError("Generated artifacts do not align with input model_list/model_id_list.")

    @staticmethod
    def _strip_think_blocks(text: str) -> str:
        s = str(text or "")
        start_tag = "<think>"
        end_tag = "</think>"
        while True:
            start = s.find(start_tag)
            if start < 0:
                break
            end = s.find(end_tag, start + len(start_tag))
            if end < 0:
                s = s[:start]
                break
            s = s[:start] + s[end + len(end_tag):]
        return s.strip()

    @staticmethod
    def _dedup_str_list(values: List[Any]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in normalize_list(values, fallback_single_str=True, stringify_items=True):
            s = str(item or "").strip()
            if not s:
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
        return out

    @staticmethod
    def _shrink_reference_text(text: str, head_chars: int = 1600, tail_chars: int = 400) -> str:
        s = str(text or "")
        if len(s) <= head_chars + tail_chars + 80:
            return s
        omitted = len(s) - head_chars - tail_chars
        return f"{s[:head_chars]}\n\n<!-- REF_TRUNCATED: omitted {omitted} chars -->\n\n{s[-tail_chars:]}"

    @staticmethod
    def _fallback_generate_from_reference(ctx: Dict[str, Any]) -> Dict[str, Any]:
        model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        family_md = str(ctx.get("ref_md") or "").strip()
        family_index_js = str(ctx.get("ref_index_js") or "").strip()
        mode = str(ctx.get("generation_mode") or "").strip().lower()

        family_md, family_index_js = GenerateReadmeTool._align_reference_family_version(
            family_md=family_md,
            family_index_js=family_index_js,
            model_list=model_list,
            model_id_list=GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or []),
        )

        if not family_md:
            family_md = "# Model Deployment\n\n"
        if model_list and not any(m.lower() in family_md.lower() for m in model_list):
            lines = [f"- {m}" for m in model_list]
            family_md = f"## Target Models\n\n{chr(10).join(lines)}\n\n{family_md}".strip()
        if is_url_source_mode(mode) and "intel" not in family_md.lower():
            family_md += "\n\n## Intel CPU\n\nAdd Intel CPU launch options aligned with CUDA/AMD structure.\n"

        if not family_index_js.strip():
            model_js = ", ".join([f'"{m}"' for m in model_list]) if model_list else ""
            family_index_js = (
                "export const modelList = ["
                + model_js
                + "];\n"
                + "export const backends = ['cuda', 'amd', 'intel_cpu'];\n"
            )
        elif is_url_source_mode(mode) and "intel" not in family_index_js.lower():
            family_index_js += "\n\n// Intel CPU backend option\n"

        return {
            "family_md": family_md.strip(),
            "family_index_js": family_index_js.strip(),
            "family_js_files": [{"path": "index.js", "content": family_index_js.strip()}],
            "source": "fallback",
        }

    @staticmethod
    def _infer_target_family_version(model_list: List[str], model_id_list: List[str]) -> tuple[str, str]:
        texts = list(model_list or []) + list(model_id_list or [])
        for raw in texts:
            t = str(raw or "")
            m = re.search(r"\b(llama|qwen|gemma|mistral|deepseek|phi)\s*[-_ ]?(\d+(?:\.\d+)?)", t, flags=re.IGNORECASE)
            if m:
                return (m.group(1).lower(), m.group(2))
        return ("", "")

    @staticmethod
    def _infer_target_size(model_list: List[str], model_id_list: List[str]) -> str:
        texts = list(model_list or []) + list(model_id_list or [])
        for raw in texts:
            t = str(raw or "")
            m = re.search(r"[-_ ](\d+(?:\.\d+)?[bBmM])(?:[-_ ]|$)", t)
            if m:
                return m.group(1).upper()
        return ""

    @staticmethod
    def _align_reference_family_version(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
    ) -> tuple[str, str]:
        fam, ver = GenerateReadmeTool._infer_target_family_version(model_list, model_id_list)
        if not fam or not ver:
            return (family_md, family_index_js)

        md = str(family_md or "")
        js = str(family_index_js or "")
        fam_title = fam.capitalize()

        if fam == "llama":
            md = re.sub(r"\bLlama\s+\d+(?:\.\d+)?\b", f"{fam_title} {ver}", md)
            md = re.sub(r"\bLlama-\d+(?:\.\d+)?\b", f"{fam_title}-{ver}", md)
            js = re.sub(r"\bLlama\s*\d+(?:\.\d+)?\b", f"{fam_title} {ver}", js)
            js = re.sub(r"\bLlama\d+(?:_\d+)?\b", f"{fam_title}{ver.replace('.', '')}", js)
        else:
            md = re.sub(
                rf"\b{fam_title}\s+\d+(?:\.\d+)?\b",
                f"{fam_title} {ver}",
                md,
                flags=re.IGNORECASE,
            )
            md = re.sub(
                rf"\b{fam_title}-\d+(?:\.\d+)?\b",
                f"{fam_title}-{ver}",
                md,
                flags=re.IGNORECASE,
            )
        return (md, js)

    @staticmethod
    def _normalize_artifacts_to_target_models(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
    ) -> tuple[str, str]:
        md, js = GenerateReadmeTool._align_reference_family_version(
            family_md=family_md,
            family_index_js=family_index_js,
            model_list=model_list,
            model_id_list=model_id_list,
        )
        fam, ver = GenerateReadmeTool._infer_target_family_version(model_list, model_id_list)
        size = GenerateReadmeTool._infer_target_size(model_list, model_id_list)
        if fam == "llama" and ver:
            ver_pat = re.escape(ver)
            # fix lingering 3.1 references in links/paths/component names
            md = re.sub(r"meta-llama-3-1", f"meta-llama-{ver.replace('.', '-')}", md)
            md = re.sub(r"llama3_1", f"llama3_{ver.replace('.', '_')}", md, flags=re.IGNORECASE)
            js = re.sub(r"Llama31", f"Llama{ver.replace('.', '')}", js)
            # keep size aligned (e.g. 8B -> 3B) when target size is explicit
            if size:
                md = re.sub(rf"(Llama-{ver_pat}-)\d+(?:\.\d+)?[bBmM]", rf"\g<1>{size}", md)
                js = re.sub(rf"(Llama-{ver_pat}-)\d+(?:\.\d+)?[bBmM]", rf"\g<1>{size}", js)
                js = re.sub(rf"(Meta-Llama-{ver_pat}-)\d+(?:\.\d+)?[bBmM]", rf"\g<1>{size}", js)
        return (md, js)

    @staticmethod
    def _ensure_readme_command_content(
        family_md: str,
        family_index_js: str,
        model_list: List[str],
        model_id_list: List[str],
    ) -> str:
        md = str(family_md or "")
        js = str(family_index_js or "")
        # Drop truncation markers if they leak into generated family artifacts.
        md = re.sub(r"\n?\s*<!--\s*REF_TRUNCATED:[\s\S]*?-->\s*\n?", "\n\n", md, flags=re.IGNORECASE)
        lowered = md.lower()
        has_cmd_block = bool(re.search(r"```(?:bash|shell|sh|console)\s*[\s\S]*?```", md, flags=re.IGNORECASE))
        has_launch_cmd = "python -m sglang.launch_server" in lowered
        has_bench_cmd = "python -m sglang.bench_serving" in lowered
        if has_cmd_block and (has_launch_cmd or has_bench_cmd):
            return md.strip()

        target_model = (
            GenerateReadmeTool._dedup_str_list(model_id_list or [])
            or GenerateReadmeTool._dedup_str_list(model_list or [])
            or ["<MODEL_ID>"]
        )[0]
        tp_match = re.search(r"--tp\s+(\d+)", js)
        tp = tp_match.group(1) if tp_match else "1"
        cmd_section = f"""
## Quick Start Commands

### Launch Serving

```bash
python -m sglang.launch_server \\
  --model-path {target_model} \\
  --host 0.0.0.0 \\
  --tp {tp}
```

### Benchmark

```bash
python -m sglang.bench_serving \\
  --dataset-name random \\
  --random-input-len 1024 \\
  --random-output-len 1024 \\
  --num-prompts 1 \\
  --max-concurrency 1 \\
  --request-rate inf
```
""".strip()
        return f"{md.rstrip()}\n\n{cmd_section}\n".strip()

    @staticmethod
    def _llm_judge_generation_mode(ctx: Dict[str, Any]) -> str:
        compact_ctx = {
            "generation_mode": str(ctx.get("generation_mode") or ""),
            "model_list": ctx.get("model_list") or [],
            "github_url": ctx.get("github_url") or [],
            "source_md_url": ctx.get("source_md_url") or "",
            "source_js_url": ctx.get("source_js_url") or "",
            "github_md_folder_url": ctx.get("github_md_folder_url") or "",
            "github_js_folder_url": ctx.get("github_js_folder_url") or "",
            "source_md_files_count": len(ctx.get("source_md_files") or []),
            "source_js_files_count": len(ctx.get("source_js_files") or []),
            "remote_payload_generation_mode": str((ctx.get("remote_payload") or {}).get("generation_mode") or ""),
            "remote_payload_source_urls": (ctx.get("remote_payload") or {}).get("source_urls") or {},
        }
        prompt = f"""
You are a strict workflow mode classifier.
Classify the generation flow as:
- "reference" (legacy md/js adaptation)
- "web_sources" (url_source/github_folders flow)

Rules:
1. If source URLs or source file collections are the primary inputs, choose "web_sources".
2. If model_list + github_url/ref_md/ref_index_js are the primary inputs, choose "reference".
3. Prefer consistency with explicit generation_mode when evidence is not conflicting.
4. Output ONLY JSON: {{"generation_mode":"reference|web_sources"}}.

Input:
{json.dumps(compact_ctx, ensure_ascii=False)}
"""
        try:
            response = GenerateReadmeTool.llm.invoke(prompt)
            cleaned = GenerateReadmeTool._strip_think_blocks(response)
            parsed = json.loads(cleaned)
            mode = str(parsed.get("generation_mode") or "").strip().lower()
            if mode in {"reference", "web_sources"}:
                return mode
        except Exception:
            pass
        return ""

    @staticmethod
    def _resolve_generation_mode(ctx: Dict[str, Any]) -> tuple[str, str]:
        explicit_mode = str(ctx.get("generation_mode") or "").strip().lower()
        remote_payload = ctx.get("remote_payload") or {}
        remote_mode = str(remote_payload.get("generation_mode") or "").strip().lower()

        model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        github_url = GenerateReadmeTool._dedup_str_list(ctx.get("github_url") or [])
        has_legacy_signal = bool(
            model_list
            or github_url
            or str(ctx.get("ref_md") or "").strip()
            or str(ctx.get("ref_index_js") or "").strip()
        )

        source_md_url = str(ctx.get("source_md_url") or "").strip()
        source_js_url = str(ctx.get("source_js_url") or "").strip()
        github_md_folder_url = str(ctx.get("github_md_folder_url") or "").strip()
        github_js_folder_url = str(ctx.get("github_js_folder_url") or "").strip()
        source_urls = remote_payload.get("source_urls") or {}
        has_source_signal = bool(
            source_md_url
            or source_js_url
            or github_md_folder_url
            or github_js_folder_url
            or str(source_urls.get("md") or "").strip()
            or str(source_urls.get("js") or "").strip()
            or len(ctx.get("source_md_files") or []) > 0
            or len(ctx.get("source_js_files") or []) > 0
            or remote_mode == "url_source"
        )

        # Strong explicit branches first.
        if explicit_mode == "reference" and has_legacy_signal and not has_source_signal:
            return ("reference", "explicit")
        if is_url_source_mode(explicit_mode) and has_source_signal and not has_legacy_signal:
            return ("web_sources", "explicit")

        # Strong data-driven branches.
        if has_source_signal and not has_legacy_signal:
            return ("web_sources", "rule")
        if has_legacy_signal and not has_source_signal:
            return ("reference", "rule")

        # Ambiguous/conflicting: let LLM arbitrate.
        judged = GenerateReadmeTool._llm_judge_generation_mode(ctx)
        if judged in {"reference", "web_sources"}:
            return (judged, "llm")

        # Conservative fallback.
        if explicit_mode == "reference":
            return ("reference", "fallback")
        if is_url_source_mode(explicit_mode):
            return ("web_sources", "fallback")
        return ("reference", "fallback")

    @staticmethod
    def _llm_generate_family_artifacts(ctx: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"""
You are generating family-level deployment docs.
Input context JSON:
{json.dumps(ctx, ensure_ascii=False)}

Output ONLY JSON with schema:
{{
  "family_md": "full markdown",
  "family_index_js": "main index.js content",
  "family_js_files": [{{"path":"index.js","content":"..."}}],
  "memory_cleanup": {{
    "model_list": ["..."]
  }}
}}

Rules:
1. One unified flow with internal branching by generation_mode.
2. legacy/reference: adapt from ref_md/ref_index_js to current model_list/model_id_list.
3. url_source/web_sources/github_folders: preserve official structure and include Intel CPU alongside CUDA/AMD.
4. Must align README and JS model choices to model_list.
5. Return valid JSON only.
"""
        response = GenerateReadmeTool.llm.invoke(prompt)
        cleaned = GenerateReadmeTool._strip_think_blocks(response)
        cleaned_lower = cleaned.lstrip().lower()
        if cleaned_lower.startswith("<!doctype html") or cleaned_lower.startswith("<html") or "ie friendly error message walkround" in cleaned_lower:
            raise RuntimeError("LLM endpoint returned HTML/proxy error page instead of JSON.")
        parsed = json.loads(cleaned)
        md = str(parsed.get("family_md") or "").strip()
        idx = str(parsed.get("family_index_js") or "").strip()
        js_files = parsed.get("family_js_files")
        if not isinstance(js_files, list):
            js_files = []
        js_files = GenerateReadmeTool._normalize_js_files(js_files)
        if not js_files and idx:
            js_files = [{"path": "index.js", "content": idx}]
        if not idx and js_files:
            primary = next((x for x in js_files if x["path"].split("/")[-1] == "index.js"), js_files[0])
            idx = str(primary.get("content") or "")
        if not md or not idx:
            raise ValueError("LLM generation returned empty family_md or family_index_js")
        models = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
        mids = GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or [])
        md, idx = GenerateReadmeTool._normalize_artifacts_to_target_models(md, idx, models, mids)
        return {
            "family_md": md,
            "family_index_js": idx,
            "family_js_files": js_files,
            "memory_cleanup": parsed.get("memory_cleanup") if isinstance(parsed.get("memory_cleanup"), dict) else {},
            "source": "llm",
        }

    @staticmethod
    def _compact_generation_memory() -> Dict[str, int]:
        memory = GenerateReadmeTool.global_memory
        model_list = GenerateReadmeTool._dedup_str_list(memory.memory_retrieve("model_list") or [])
        model_id_list = GenerateReadmeTool._dedup_str_list(memory.memory_retrieve("model_id_list") or [])
        model_url_list = GenerateReadmeTool._dedup_str_list(memory.memory_retrieve("model_url_list") or [])
        github_url = normalize_list(memory.memory_retrieve("github_url") or [], fallback_single_str=True, stringify_items=True)

        if model_list:
            memory.memory_store("model_list", model_list)
        if model_id_list:
            memory.memory_store("model_id_list", model_id_list)
        if model_url_list:
            memory.memory_store("model_url_list", model_url_list)
        if model_list:
            if len(github_url) < len(model_list):
                github_url = github_url + [""] * (len(model_list) - len(github_url))
            elif len(github_url) > len(model_list):
                github_url = github_url[: len(model_list)]
            memory.memory_store("github_url", github_url)

        # Keep reference only as lightweight structure guidance after final artifacts are generated.
        family_md = str(memory.memory_retrieve("family_md") or "")
        family_index_js = str(memory.memory_retrieve("family_index_js") or "")
        if family_md.strip() and family_index_js.strip():
            ref_md = str(memory.memory_retrieve("ref_md") or "")
            ref_js = str(memory.memory_retrieve("ref_index_js") or "")
            if ref_md:
                memory.memory_store("ref_md", GenerateReadmeTool._shrink_reference_text(ref_md))
            if ref_js:
                memory.memory_store("ref_index_js", GenerateReadmeTool._shrink_reference_text(ref_js, head_chars=1200, tail_chars=300))
            rp = memory.memory_retrieve("remote_payload") or {}
            if isinstance(rp, dict) and str(rp.get("generation_mode") or "").strip().lower() == "legacy":
                rp2 = dict(rp)
                rp2["content"] = {"from_memory": True}
                memory.memory_store("remote_payload", rp2)
        return {
            "model_list": len(model_list),
            "model_id_list": len(model_id_list),
            "model_url_list": len(model_url_list),
            "github_url": len(github_url),
        }

    @tool("memory_retrieve_generation_context")
    def memory_retrieve_generation_context():
        """Retrieve generation context from GLOBAL_MEMORY for canonical family_content generation."""
        return {
            "generation_mode": GenerateReadmeTool.global_memory.memory_retrieve("generation_mode") or "reference",
            "remote_payload": GenerateReadmeTool.global_memory.memory_retrieve("remote_payload") or {},
            "github_md_folder_url": GenerateReadmeTool.global_memory.memory_retrieve("github_md_folder_url") or "",
            "github_js_folder_url": GenerateReadmeTool.global_memory.memory_retrieve("github_js_folder_url") or "",
            "model_list": GenerateReadmeTool.global_memory.memory_retrieve("model_list") or [],
            "model_id_list": GenerateReadmeTool.global_memory.memory_retrieve("model_id_list") or [],
            "model_url_list": GenerateReadmeTool.global_memory.memory_retrieve("model_url_list") or [],
            "github_url": GenerateReadmeTool.global_memory.memory_retrieve("github_url") or [],
            "ref_md": GenerateReadmeTool.global_memory.memory_retrieve("ref_md") or "",
            "ref_index_js": GenerateReadmeTool.global_memory.memory_retrieve("ref_index_js") or "",
            "source_md_files": GenerateReadmeTool.global_memory.memory_retrieve("source_md_files") or [],
            "source_js_files": GenerateReadmeTool.global_memory.memory_retrieve("source_js_files") or [],
            "family_md": GenerateReadmeTool.global_memory.memory_retrieve("family_md") or "",
            "family_index_js": GenerateReadmeTool.global_memory.memory_retrieve("family_index_js") or "",
            "family_js_files": GenerateReadmeTool.global_memory.memory_retrieve("family_js_files") or [],
            "family_content": GenerateReadmeTool.global_memory.memory_retrieve("family_content") or "",
        }

    @tool("memory_generate_and_store_family_artifacts")
    def memory_generate_and_store_family_artifacts() -> Dict[str, Any]:
        """Generate family_md/index.js from ref/source + model_list, store artifacts, and compact redundant memory lists."""
        ctx = GenerateReadmeTool.memory_retrieve_generation_context.func()
        resolved_mode, mode_from = GenerateReadmeTool._resolve_generation_mode(ctx)
        ctx["generation_mode"] = resolved_mode
        GenerateReadmeTool.global_memory.memory_store("generation_mode", resolved_mode)
        generated: Dict[str, Any]
        llm_error = ""
        try:
            generated = GenerateReadmeTool._llm_generate_family_artifacts(ctx)
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            generated = GenerateReadmeTool._fallback_generate_from_reference(ctx)

        def _store_from_generated(g: Dict[str, Any]) -> Dict[str, Any]:
            family_md = str(g.get("family_md") or "").strip()
            family_index_js = str(g.get("family_index_js") or "").strip()
            model_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_list") or [])
            model_id_list = GenerateReadmeTool._dedup_str_list(ctx.get("model_id_list") or [])
            family_md_norm, family_index_js_norm = GenerateReadmeTool._normalize_artifacts_to_target_models(
                family_md,
                family_index_js,
                model_list,
                model_id_list,
            )
            family_md_norm = GenerateReadmeTool._ensure_readme_command_content(
                family_md=family_md_norm,
                family_index_js=family_index_js_norm,
                model_list=model_list,
                model_id_list=model_id_list,
            )
            js_files = GenerateReadmeTool._normalize_js_files(g.get("family_js_files") or [])
            if not js_files and family_index_js_norm:
                js_files = [{"path": "index.js", "content": family_index_js_norm}]
            if len(js_files) > 1:
                return GenerateReadmeTool.memory_store_family_multi_artifacts.func(
                    family_md=family_md_norm,
                    family_js_files_json=json.dumps(js_files, ensure_ascii=False),
                )
            return GenerateReadmeTool.memory_store_family_artifacts.func(
                family_md=family_md_norm,
                family_index_js=family_index_js_norm,
            )

        try:
            store_result = _store_from_generated(generated)
        except Exception:
            generated = GenerateReadmeTool._fallback_generate_from_reference(ctx)
            store_result = _store_from_generated(generated)

        compacted = GenerateReadmeTool._compact_generation_memory()
        debug_info = {
            "resolved_generation_mode": resolved_mode,
            "mode_decision_source": mode_from,
            "generation_source": generated.get("source", "unknown"),
            "llm_error": llm_error,
        }
        GenerateReadmeTool.global_memory.memory_store("readme_generation_debug", debug_info)
        return {
            "ok": True,
            "resolved_generation_mode": resolved_mode,
            "mode_decision_source": mode_from,
            "generation_source": generated.get("source", "unknown"),
            "llm_error": llm_error,
            "store_result": store_result,
            "compacted_lengths": compacted,
        }

    @tool("memory_store_family_content")
    def memory_store_family_content(family_content: str):
        """Store canonical family content (single merged md+js content) into GLOBAL_MEMORY with key "family_content"."""
        content = family_content or ""
        md_text = content
        js_text = ""
        js_match = re.search(r"```javascript\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
        if js_match:
            js_text = js_match.group(1).strip()
            md_text = (content[: js_match.start()] + content[js_match.end() :]).strip()
        GenerateReadmeTool._validate_target_models(md_text, js_text)
        GenerateReadmeTool.global_memory.memory_store("family_content", content)
        GenerateReadmeTool.global_memory.memory_store("family_md", md_text)
        GenerateReadmeTool.global_memory.memory_store("family_index_js", js_text)
        return {"ok": True, "family_md_length": len(md_text), "family_index_js_length": len(js_text)}

    @tool("memory_store_family_artifacts")
    def memory_store_family_artifacts(family_md: str, family_index_js: str):
        """Store family README.md + index.js artifacts and compose canonical family_content."""
        GenerateReadmeTool._validate_target_models(family_md or "", family_index_js or "")
        GenerateReadmeTool.global_memory.memory_store("family_md", family_md or "")
        GenerateReadmeTool.global_memory.memory_store("family_index_js", family_index_js or "")
        js_files = [{"path": "index.js", "content": family_index_js or ""}]
        GenerateReadmeTool.global_memory.memory_store("family_js_files", js_files)
        family_content = GenerateReadmeTool._compose_family_content(
            family_md or "",
            family_index_js or "",
            js_files,
        )
        GenerateReadmeTool.global_memory.memory_store("family_content", family_content)
        return {
            "ok": True,
            "family_md_length": len(family_md or ""),
            "family_index_js_length": len(family_index_js or ""),
            "family_content_length": len(family_content or ""),
        }

    @tool("memory_store_family_multi_artifacts")
    def memory_store_family_multi_artifacts(family_md: str, family_js_files_json: str):
        """Store family README.md plus multiple JS files; keeps index.js as primary for compatibility."""
        js_files_raw = []
        try:
            js_files_raw = family_js_files_json if isinstance(family_js_files_json, list) else re.sub(r"^```json|```$", "", str(family_js_files_json).strip(), flags=re.IGNORECASE).strip()
            if isinstance(js_files_raw, str):
                js_files_raw = js_files_raw or "[]"
                js_files_raw = json.loads(js_files_raw)
        except Exception as e:
            raise ValueError(f"family_js_files_json must be valid JSON list: {e}")

        js_files = GenerateReadmeTool._normalize_js_files(js_files_raw)
        index_item = next((x for x in js_files if x["path"].split("/")[-1] == "index.js"), js_files[0] if js_files else {"content": ""})
        family_index_js = str(index_item.get("content") or "")
        GenerateReadmeTool._validate_target_models(family_md or "", family_index_js or "")

        GenerateReadmeTool.global_memory.memory_store("family_md", family_md or "")
        GenerateReadmeTool.global_memory.memory_store("family_index_js", family_index_js)
        GenerateReadmeTool.global_memory.memory_store("family_js_files", js_files)
        family_content = GenerateReadmeTool._compose_family_content(family_md or "", family_index_js or "", js_files)
        GenerateReadmeTool.global_memory.memory_store("family_content", family_content)
        return {"ok": True, "js_file_count": len(js_files), "family_content_length": len(family_content)}
