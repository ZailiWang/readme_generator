#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from readme_generator.tools.model_search_tool import HuggingFaceModelClient  # noqa: E402


def main() -> int:
    client = HuggingFaceModelClient()

    def fake_search(_query: str, limit: int = 40):
        _ = limit
        return [
            {"modelId": "psychopenguin/indian_legal_llama3.2-3b-instruct"},
            {"modelId": "meta-llama/Llama-3.2-3B-Instruct"},
            {"modelId": "RedHatAI/Llama-3.2-3B-Instruct-FP8"},
            {"modelId": "hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4"},
        ]

    client._search = fake_search  # type: ignore[method-assign]

    result = client.batch_search(
        [
            "Llama-3.2-3B-Instruct",
            "Llama-3.2-3B-Instruct-FP8",
            "Llama-3.2-3B-Instruct-AWQ",
        ]
    )
    mids = [str(x).lower() for x in (result.get("model_id_list") or [])]
    assert any("meta-llama/llama-3.2-3b-instruct" in x for x in mids)
    assert any("fp8" in x for x in mids)
    assert any("awq" in x for x in mids)
    assert len(mids) >= 3
    print("PASS: model_search keeps explicit variants and avoids over-collapse")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

