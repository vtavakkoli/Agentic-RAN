from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML report from metrics and predictions")
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", default="results/final/report.html")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    preds = pd.read_csv(args.preds)

    summary = {
        "rows": len(preds),
        "mae": float((preds["error"].abs()).mean()),
        "rmse": float(((preds["error"] ** 2).mean()) ** 0.5),
        "mean_true": float(preds["y_true"].mean()),
        "mean_pred": float(preds["y_pred"].mean()),
    }

    html = f"""
    <html><head><title>Agentic-RAN Report</title></head>
    <body>
    <h1>Training & Prediction Report</h1>
    <h2>Metrics JSON</h2>
    <pre>{json.dumps(metrics, indent=2)}</pre>
    <h2>Prediction Summary</h2>
    <pre>{json.dumps(summary, indent=2)}</pre>
    <h2>Prediction Samples</h2>
    {preds.head(100).to_html(index=False)}
    </body></html>
    """
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Saved report: {out}")


if __name__ == "__main__":
    main()
