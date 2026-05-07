import json
import re
from typing import Any, List

MODEL_FAMILIES = ("qwen", "llama", "gemma", "mistral", "deepseek", "phi")


def is_url_source_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in {"web_sources", "github_folders", "url_source"}


def normalize_list(
    value: Any,
    *,
    fallback_single_str: bool = False,
    stringify_items: bool = False,
) -> List[Any]:
    """Normalize arbitrary input into a list.

    - list -> list
    - json-string(list) -> parsed list
    - plain string -> [] or [string] based on fallback_single_str
    - others -> []
    """
    items: List[Any]
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:
            parsed = json.loads(value)
            items = parsed if isinstance(parsed, list) else ([value] if fallback_single_str else [])
        except Exception:
            items = [value] if fallback_single_str else []
    else:
        items = []

    if not stringify_items:
        return items
    return [str(v) if v is not None else "" for v in items]


def infer_models_from_text(text: str) -> List[str]:
    if not text:
        return []
    blacklist = {
        "github.com",
        "docs.sglang.ai",
        "site/src",
        "docs/autoregressive",
        "components/autoregressive",
        "sglang/tree",
        "sglang/blob",
        "main/docs",
        "main/src",
    }
    model_keywords = MODEL_FAMILIES
    candidates: List[str] = []

    for m in re.findall(r"\b[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:-[A-Za-z0-9_.-]+)*\b", text):
        ml = m.lower()
        if any(b in ml for b in blacklist):
            continue
        if "http" in ml:
            continue
        left, right = m.split("/", 1)
        if right.lower().endswith(".md") or "configgenerator" in right.lower():
            continue
        if any(k in right.lower() for k in model_keywords) or any(k in left.lower() for k in model_keywords):
            candidates.append(m)

    for m in re.findall(
        r"\b(?:Qwen|Llama|Gemma|Mistral|DeepSeek|Phi)[A-Za-z0-9_.-]*\b",
        text,
        flags=re.IGNORECASE,
    ):
        ml = m.lower()
        if ml.endswith(".md") or "configgenerator" in ml:
            continue
        if len(m) >= 4:
            candidates.append(m)

    # Family + size patterns from docs/index.js, e.g. "Qwen3 7B", "Qwen 3 14B Instruct"
    family_size_pattern = re.compile(
        r"\b(?P<family>Qwen|Llama|Gemma|Mistral|DeepSeek|Phi)\s*"
        r"(?P<version>\d+(?:\.\d+)?)?\s*[-_ ]*\s*"
        r"(?P<size>\d+(?:\.\d+)?\s*[BMbm])"
        r"(?:\s*[-_ ]\s*(?P<variant>Instruct|Chat|Base|FP8|AWQ|INT4|INT8|W8A8|Quantized))?\b",
        flags=re.IGNORECASE,
    )
    for m in family_size_pattern.finditer(text):
        family = m.group("family") or ""
        version = (m.group("version") or "").strip()
        size = (m.group("size") or "").replace(" ", "")
        variant = (m.group("variant") or "").strip()
        if not family or not size:
            continue
        base = f"{family}{version}-{size}" if version else f"{family}-{size}"
        candidates.append(base)
        if variant:
            candidates.append(f"{base}-{variant}")

    # Quoted literals in JS/TS often hold model ids or names.
    for lit in re.findall(r"['\"]([^'\"]{3,120})['\"]", text):
        ll = lit.lower()
        if any(k in ll for k in model_keywords) and re.search(r"\d+(?:\.\d+)?\s*[bm]", ll):
            candidates.append(lit.strip())

    out: List[str] = []
    seen = set()
    for c in candidates:
        k = c.strip()
        if not k:
            continue
        lk = k.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(k)
    return out


def infer_family_hint_from_corpus(corpus: List[str]) -> str:
    scores = {k: 0 for k in MODEL_FAMILIES}
    for text in corpus:
        t = str(text or "").lower()
        for fam in MODEL_FAMILIES:
            scores[fam] += t.count(fam)
    best = max(scores, key=scores.get)
    return best if scores.get(best, 0) > 0 else ""


def infer_family_anchor_from_corpus(corpus: List[str], family_hint: str = "") -> str:
    fam = (family_hint or infer_family_hint_from_corpus(corpus) or "").strip().lower()
    if not fam:
        return ""
    joined = "\n".join(str(x or "") for x in corpus)
    patterns = {
        "qwen": r"\b(Qwen\s*[-_ ]?\s*\d+(?:\.\d+)?)\b",
        "llama": r"\b(Llama\s*[-_ ]?\s*\d+(?:\.\d+)?)\b",
        "gemma": r"\b(Gemma\s*[-_ ]?\s*\d+(?:\.\d+)?)\b",
        "mistral": r"\b(Mistral\s*[-_ ]?\s*\d+(?:\.\d+)?)\b",
        "deepseek": r"\b(DeepSeek\s*[-_ ]?\s*\d+(?:\.\d+)?)\b",
        "phi": r"\b(Phi\s*[-_ ]?\s*\d+(?:\.\d+)?)\b",
    }
    p = patterns.get(fam)
    if p:
        m = re.search(p, joined, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            raw = re.sub(r"[\s_-]+", "", raw)
            return raw
    return fam.capitalize()


def extract_label_sizes_from_text(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out
    patterns = [
        r"label\s*[:=]\s*['\"]\s*(\d+(?:\.\d+)?)\s*([bBmM])\s*['\"]",
        r"['\"]label['\"]\s*:\s*['\"]\s*(\d+(?:\.\d+)?)\s*([bBmM])\s*['\"]",
    ]
    seen = set()
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            size = f"{m.group(1)}{m.group(2).upper()}"
            if size.lower() in seen:
                continue
            seen.add(size.lower())
            out.append(size)
    return out


def _is_candidate_noise(token: str) -> bool:
    t = token.lower()
    noise_substrings = (
        "http://",
        "https://",
        "github.com",
        "huggingface.co",
        "/blob/",
        "/tree/",
        "blob/",
        "tree/",
        "docs/",
        "src/",
        "components/",
        "autoregressive/",
    )
    return any(x in t for x in noise_substrings)


def _looks_like_concrete_model(token: str) -> bool:
    t = token.lower()
    has_size = bool(re.search(r"\d+(?:\.\d+)?\s*[bm]", t))
    has_variant = any(k in t for k in ("instruct", "chat", "fp8", "awq", "int4", "int8", "w8a8", "quantized"))
    return has_size or has_variant


def filter_model_candidates(candidates: List[str], family_hint: str = "") -> List[str]:
    fam = (family_hint or "").strip().lower()
    out: List[str] = []
    seen = set()
    for token in candidates:
        raw = str(token or "").strip()
        if not raw:
            continue
        low = raw.lower()
        if _is_candidate_noise(low):
            continue
        if fam and fam not in low:
            continue
        if "/" in raw:
            left, right = raw.split("/", 1)
            if not left.strip() or not right.strip():
                continue
            if _is_candidate_noise(left) or _is_candidate_noise(right):
                continue
            if not _looks_like_concrete_model(right):
                continue
        else:
            if not _looks_like_concrete_model(raw):
                continue
        if low in seen:
            continue
        seen.add(low)
        out.append(raw)
    return out


def infer_models_from_corpus(corpus: List[str], family_hint: str = "") -> List[str]:
    inferred: List[str] = []
    for text in corpus:
        inferred.extend(infer_models_from_text(str(text or "")))
    fam = (family_hint or infer_family_hint_from_corpus(corpus) or "").strip().lower()
    family_anchor = infer_family_anchor_from_corpus(corpus, family_hint=fam)
    if family_anchor:
        label_sizes: List[str] = []
        for text in corpus:
            label_sizes.extend(extract_label_sizes_from_text(str(text or "")))
        for size in label_sizes:
            inferred.append(f"{family_anchor}-{size}")
    inferred = filter_model_candidates(inferred, family_hint=fam)
    final: List[str] = []
    seen = set()
    for m in inferred:
        lk = m.lower()
        if lk in seen:
            continue
        seen.add(lk)
        final.append(m)
    return final
