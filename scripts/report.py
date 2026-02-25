from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _scenario_status_table() -> pd.DataFrame:
    status_files = sorted(Path("results/scenarios").glob("*/status.json"))
    rows = []
    for f in status_files:
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            rows.append({"scenario_name": f.parent.name, "success": False, "error": "invalid status.json"})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate HTML report")
    p.add_argument("--preds", required=True)
    p.add_argument("--metrics", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=None)
    args = p.parse_args()

    preds = pd.read_csv(args.preds)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    cfg = {}
    if args.config and Path(args.config).exists():
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    fig1, ax1 = plt.subplots(figsize=(8, 3))
    ax1.plot(preds["y_true"].values[:300], label="y_true")
    ax1.plot(preds["y_pred"].values[:300], label="y_pred")
    ax1.legend()
    ax1.set_title("y_true vs y_pred")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.hist(preds["error"].dropna(), bins=30)
    ax2.set_title("Residual histogram")

    img1 = _fig_to_base64(fig1)
    img2 = _fig_to_base64(fig2)
    plt.close(fig1)
    plt.close(fig2)

    status_df = _scenario_status_table()
    status_html = status_df.to_html(index=False) if not status_df.empty else "<p>No scenario statuses found.</p>"

    html = f"""
    <html><body>
    <h1>KPM Final Report</h1>
    <h2>Metrics (val/test)</h2>
    <pre>{json.dumps(metrics, indent=2)}</pre>
    <h2>Run config summary</h2>
    <pre>{json.dumps(cfg, indent=2)}</pre>
    <h2>Predictions</h2>
    <img src='data:image/png;base64,{img1}'/>
    <h2>Residuals</h2>
    <img src='data:image/png;base64,{img2}'/>
    <h2>Scenario status</h2>
    {status_html}
    </body></html>
    """

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"report written to {out}")


if __name__ == "__main__":
    main()
