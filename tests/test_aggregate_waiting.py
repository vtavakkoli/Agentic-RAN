from __future__ import annotations

import json

from scripts.aggregate_report import _status_ready


def test_status_ready_requires_end_time(tmp_path) -> None:
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"success": True}), encoding="utf-8")
    assert _status_ready(p) is False

    p.write_text(json.dumps({"success": True, "end_time": "2024-01-01T00:00:00Z"}), encoding="utf-8")
    assert _status_ready(p) is True
