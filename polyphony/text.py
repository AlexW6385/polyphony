"""Small text helpers shared by voices, frames and ensembles."""

from __future__ import annotations

from typing import Any

import yaml


def parse_front_matter(raw: str) -> tuple[str, dict[str, Any]]:
    """Split optional YAML front matter from the body. The body is free text."""
    raw = raw.lstrip("﻿")
    lines = raw.splitlines()
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                meta = yaml.safe_load("\n".join(lines[1:i])) or {}
                if not isinstance(meta, dict):
                    raise ValueError("front matter must be a mapping")
                return "\n".join(lines[i + 1 :]).strip(), meta
    return raw.strip(), {}
