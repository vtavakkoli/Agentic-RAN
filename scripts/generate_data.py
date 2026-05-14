from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from agentic_ran.data_loading import DEFAULT_FEATURES, DEFAULT_TARGET_COL
from scripts.prepare_splits import build_dataset, split_and_save


def _html_table_from_dict(title: str, data: dict) -> str:
    rows = "".join(
        f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in data.items()
    )
    return f"<h2>{title}</h2><table>{rows}</table>"


def _write_report(summary: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_rows = summary.get("split", {}).get("rows", {})
    preprocessing = summary.get("preprocessing", {})
    files_used = preprocessing.get("files_used", [])

    split_table = pd.DataFrame(
        [{"split": k, "rows": v} for k, v in split_rows.items()]
    ).to_html(index=False, escape=False)
    file_table = pd.DataFrame(
        [{"file": f} for f in files_used[:50]]
    ).to_html(index=False, escape=False) if files_used else "<p>No source files listed.</p>"

    html = f"""
<html>
<head>
<meta charset='utf-8'>
<title>Generate Data Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;background:#f8fafc;color:#0f172a;}}
h1,h2{{color:#0b3a75;}} table{{border-collapse:collapse;width:100%;background:white;margin:10px 0 22px 0;}}
th,td{{border:1px solid #cbd5e1;padding:8px;font-size:13px;text-align:left;}} th{{background:#dbeafe;color:#1e3a8a;}}
.section{{background:white;border:1px solid #cbd5e1;border-radius:10px;padding:14px 18px;margin-bottom:16px;}}
code{{background:#e2e8f0;padding:2px 5px;border-radius:4px;}}
</style>
</head>
<body>
<h1>Dataset Generation Report</h1>
<div class='section'>
<p><strong>Output dataset folder:</strong> <code>shared_data/splits</code></p>
<p>The command creates the three required datasets for the pipeline: <code>train.csv</code>, <code>val.csv</code> / verification, and <code>test.csv</code>.</p>
</div>
<div class='section'>
{_html_table_from_dict('Preprocessing summary', {
    'metrics files used': preprocessing.get('metrics_file_count', 0),
    'target column': preprocessing.get('target_column', ''),
    'rows per source file limit': preprocessing.get('rows_per_file', ''),
    'max files': preprocessing.get('max_files', ''),
    'keep zero requested PRBs': preprocessing.get('keep_zero_requested_prbs', False),
})}
</div>
<div class='section'>
<h2>Split rows</h2>
{split_table}
</div>
<div class='section'>
<h2>First source files used</h2>
{file_table}
</div>
</body></html>
"""
    output_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/verification/test dataset splits.")
    parser.add_argument("--max-files", type=int, default=int(os.getenv("PREP_MAX_FILES", "240")))
    parser.add_argument("--rows-per-file", type=int, default=int(os.getenv("PREP_ROWS_PER_FILE", "300")))
    parser.add_argument("--feature-col", action="append", dest="feature_cols", default=None)
    parser.add_argument("--target-col", default=DEFAULT_TARGET_COL)
    parser.add_argument("--keep-zero-requested-prbs", action="store_true")
    args = parser.parse_args()

    input_dirs = [Path("dataset/slice_mixed"), Path("dataset/slice_traffic"), Path("dataset")]
    per_file_frames, prep_summary = build_dataset(
        input_dirs=input_dirs,
        max_files=args.max_files,
        rows_per_file=args.rows_per_file,
        selected_features=args.feature_cols or DEFAULT_FEATURES,
        target_col=args.target_col,
        keep_zero_requested_prbs=args.keep_zero_requested_prbs,
    )
    split_summary = split_and_save(per_file_frames, output_dir=Path("shared_data/splits"))

    full_summary = {
        "inputs": [str(p) for p in input_dirs],
        "preprocessing": prep_summary,
        "split": split_summary,
    }
    summary_path = Path("shared_data/splits/summary.json")
    summary_path.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")

    report_path = Path("results/generate_data/report.html")
    _write_report(full_summary, report_path)
    print("[generate_data] Dataset prepared at shared_data/splits")
    print(f"[generate_data] report saved to {report_path}")


if __name__ == "__main__":
    main()
