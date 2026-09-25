"""License center helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from .constants import COMMERCIAL_NOTICE, LICENSE_NOTICE, RESPONSIBLE_USE_NOTICE


def find_license_text(start: str | Path) -> str:
    root = Path(start).resolve()
    candidates = [root / "LICENSE", root / "LICENSE.md", root.parent / "LICENSE", root.parent / "LICENSE.md"]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            return text[:4000]
    return "Please refer to the LICENSE file in this repository for the code license."


def allowed_usage_summary() -> Dict[str, str]:
    return {
        "Personal research": "yes",
        "Internal evaluation": "depends on model license",
        "Commercial production": "requires commercial model license",
        "Redistribution": "requires explicit permission",
        "SaaS / API usage": "requires commercial agreement",
        "Face swap commercial usage": "requires commercial agreement",
    }


def license_summary_text(status: str, model_name: str, provider: str, workspace: str) -> str:
    lines = [
        "# InsightFace License Summary",
        "",
        f"- Current status: {status}",
        f"- Model: {model_name}",
        f"- Provider: {provider}",
        f"- Workspace: {workspace}",
        "",
        LICENSE_NOTICE,
        COMMERCIAL_NOTICE,
        RESPONSIBLE_USE_NOTICE,
        "This tool does not provide legal advice.",
    ]
    return "\n".join(lines)
