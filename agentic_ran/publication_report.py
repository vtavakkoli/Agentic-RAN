"""Self-contained HTML reporting for the final COMMAG publication benchmark."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _esc(value)
    if not np.isfinite(number):
        return "—"
    if abs(number) < 0.001 and number != 0:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _architecture_svg() -> str:
    nodes = [
        (20, 72, 150, 58, "COMMAG", "Pinned full traces"),
        (215, 18, 170, 58, "Policy fit", "disjoint configs"),
        (215, 92, 170, 58, "Independent OPE", "disjoint configs"),
        (430, 18, 180, 58, "Validation", "thresholds + hysteresis"),
        (430, 92, 180, 58, "Seen / unseen test", "never used for tuning"),
        (655, 18, 185, 58, "Agentic controller", "critics + OOD + planning"),
        (655, 92, 185, 58, "Independent scoring", "OPE + clustered stats"),
        (885, 55, 165, 58, "Publication evidence", "HTML + CSV + JSON"),
    ]
    lines = [
        (170, 101, 215, 47), (170, 101, 215, 121),
        (385, 47, 430, 47), (385, 121, 430, 121),
        (610, 47, 655, 47), (610, 121, 655, 121),
        (840, 47, 885, 84), (840, 121, 885, 84),
    ]
    parts = [
        '<svg viewBox="0 0 1070 175" role="img" aria-label="Publication benchmark architecture">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>',
    ]
    for x1, y1, x2, y2 in lines:
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
    for x, y, w, h, title, subtitle in nodes:
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#f8fafc" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x+w/2}" y="{y+23}" text-anchor="middle" font-size="13" font-weight="700" fill="#0f172a">{_esc(title)}</text>')
        parts.append(f'<text x="{x+w/2}" y="{y+42}" text-anchor="middle" font-size="10.5" fill="#475569">{_esc(subtitle)}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _bar_chart(frame: pd.DataFrame, metric: str, title: str, lower_is_better: bool = False) -> str:
    if frame.empty or metric not in frame:
        return "<p class='muted'>No data.</p>"
    data = frame[["split", "model", metric]].dropna().copy()
    if data.empty:
        return "<p class='muted'>No data.</p>"
    models = list(dict.fromkeys(data["model"].astype(str)))
    splits = list(dict.fromkeys(data["split"].astype(str)))
    values = data[metric].astype(float).to_numpy()
    vmax = max(float(np.max(values)), 1e-9)
    vmin = min(0.0, float(np.min(values)))
    width, height = 760, 310
    left, top, bottom = 60, 34, 55
    plot_h = height - top - bottom
    group_w = (width - left - 20) / max(len(splits), 1)
    bar_w = min(42.0, group_w / max(len(models) + 1, 2))
    palette = ["#2563eb", "#0f766e", "#7c3aed", "#c2410c", "#475569", "#be123c"]
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title)}">']
    parts.append(f'<text x="{left}" y="18" font-size="13" font-weight="700" fill="#0f172a">{_esc(title)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-15}" y2="{top+plot_h}" stroke="#cbd5e1"/>')
    for si, split in enumerate(splits):
        subset = data[data["split"].astype(str) == split].set_index("model")
        x0 = left + si * group_w + 20
        for mi, model in enumerate(models):
            if model not in subset.index:
                continue
            val = float(subset.loc[model, metric])
            normalized = (val - vmin) / max(vmax - vmin, 1e-9)
            bh = max(1.0, normalized * plot_h)
            x = x0 + mi * bar_w
            y = top + plot_h - bh
            color = palette[mi % len(palette)]
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-5:.1f}" height="{bh:.1f}" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{x+(bar_w-5)/2:.1f}" y="{max(y-4,12):.1f}" text-anchor="middle" font-size="8.5" fill="#334155">{_fmt(val,3)}</text>')
        parts.append(f'<text x="{x0+(len(models)*bar_w)/2:.1f}" y="{height-32}" text-anchor="middle" font-size="11" fill="#334155">{_esc(split)}</text>')
    legend_x = left
    for mi, model in enumerate(models):
        x = legend_x + (mi % 3) * 220
        y = height - 12 - (mi // 3) * 16
        parts.append(f'<rect x="{x}" y="{y-9}" width="10" height="10" rx="2" fill="{palette[mi % len(palette)]}"/>')
        parts.append(f'<text x="{x+15}" y="{y}" font-size="9.5" fill="#475569">{_esc(model)}</text>')
    direction = "lower is better" if lower_is_better else "higher is better"
    parts.append(f'<text x="{width-18}" y="18" text-anchor="end" font-size="9" fill="#64748b">{direction}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _forest_plot(stats: dict[str, Any]) -> str:
    rows = []
    for name, item in stats.items():
        if not isinstance(item, dict) or "mean_cluster_delta_utility" not in item:
            continue
        rows.append((name, float(item["mean_cluster_delta_utility"]), float(item["bootstrap_ci95_low"]), float(item["bootstrap_ci95_high"]), float(item.get("cohens_dz", 0.0))))
    if not rows:
        return "<p class='muted'>No clustered comparison data.</p>"
    lo = min(r[2] for r in rows)
    hi = max(r[3] for r in rows)
    span = max(hi - lo, 1e-6)
    lo -= span * 0.15
    hi += span * 0.15
    width = 820
    row_h = 38
    height = 55 + row_h * len(rows)
    x0, x1 = 260, width - 35
    def sx(v: float) -> float:
        return x0 + (v - lo) / (hi - lo) * (x1 - x0)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Cluster-level effect forest plot">']
    parts.append('<text x="12" y="20" font-size="13" font-weight="700" fill="#0f172a">Cluster-level utility effects (proposed − baseline)</text>')
    zero = sx(0.0)
    parts.append(f'<line x1="{zero:.1f}" y1="32" x2="{zero:.1f}" y2="{height-18}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
    for i, (name, mean, low, high, dz) in enumerate(rows):
        y = 48 + i * row_h
        parts.append(f'<text x="12" y="{y+4}" font-size="10.5" fill="#334155">{_esc(name)}</text>')
        parts.append(f'<line x1="{sx(low):.1f}" y1="{y}" x2="{sx(high):.1f}" y2="{y}" stroke="#334155" stroke-width="2"/>')
        parts.append(f'<circle cx="{sx(mean):.1f}" cy="{y}" r="5" fill="#2563eb"/>')
        parts.append(f'<text x="{x1}" y="{y-7}" text-anchor="end" font-size="9" fill="#64748b">Δ={mean:.4f}, dz={dz:.2f}</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _ood_chart(ood: dict[str, Any]) -> str:
    scenarios = []
    for split_name in ("seen", "unseen"):
        for scenario, item in ood.get(split_name, {}).get("by_scenario", {}).items():
            scenarios.append((scenario, float(item.get("mean_ood", 0)), float(item.get("p95_ood", 0))))
    if not scenarios:
        return "<p class='muted'>No OOD scenario data.</p>"
    width, height = 760, 300
    left, top, bottom = 62, 38, 60
    plot_h = height - top - bottom
    group_w = (width-left-20)/len(scenarios)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="OOD score by scenario">']
    parts.append('<text x="12" y="20" font-size="13" font-weight="700" fill="#0f172a">OOD score by scenario</text>')
    for i,(name,mean,p95) in enumerate(scenarios):
        x = left + i*group_w + group_w*0.22
        mw = group_w*0.24
        h1 = mean*plot_h
        h2 = p95*plot_h
        parts.append(f'<rect x="{x:.1f}" y="{top+plot_h-h1:.1f}" width="{mw:.1f}" height="{h1:.1f}" rx="3" fill="#0f766e"/>')
        parts.append(f'<rect x="{x+mw+5:.1f}" y="{top+plot_h-h2:.1f}" width="{mw:.1f}" height="{h2:.1f}" rx="3" fill="#7c3aed"/>')
        label = name.replace("rome_", "")
        parts.append(f'<text x="{x+mw:.1f}" y="{height-35}" text-anchor="middle" font-size="9" fill="#475569">{_esc(label)}</text>')
    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-15}" y2="{top+plot_h}" stroke="#cbd5e1"/>')
    parts.append('<rect x="570" y="12" width="10" height="10" fill="#0f766e"/><text x="585" y="21" font-size="9.5" fill="#475569">mean</text>')
    parts.append('<rect x="630" y="12" width="10" height="10" fill="#7c3aed"/><text x="645" y="21" font-size="9.5" fill="#475569">p95</text>')
    parts.append('</svg>')
    return ''.join(parts)


def _table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 12) -> str:
    if frame is None or frame.empty:
        return "<p class='muted'>No rows.</p>"
    data = frame.copy()
    if columns:
        data = data[[c for c in columns if c in data.columns]]
    data = data.head(max_rows)
    head = ''.join(f'<th>{_esc(c)}</th>' for c in data.columns)
    body = []
    for _, row in data.iterrows():
        cells = ''.join(f'<td>{_fmt(v) if isinstance(v,(float,np.floating)) else _esc(v)}</td>' for v in row)
        body.append(f'<tr>{cells}</tr>')
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def render_report(
    output: str | Path,
    *,
    summary: dict[str, Any],
    baselines: pd.DataFrame,
    clustered_statistics: dict[str, Any],
    ood: dict[str, Any],
    ood_detection: dict[str, Any],
    transition_audit: pd.DataFrame,
    shortcut: pd.DataFrame,
    calibration: pd.DataFrame,
    latency: dict[str, Any],
    literature_reference: dict[str, Any],
) -> Path:
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    report = destination / "report.html"
    readiness = summary.get("evidence_status", "UNKNOWN")
    gates = summary.get("readiness_gates", {})
    gate_html = ''.join(
        f'<li class="{ "ok" if passed else "bad" }"><span>{"✓" if passed else "✕"}</span>{_esc(name)}</li>'
        for name, passed in gates.items()
    )
    warnings = ''.join(f'<li>{_esc(item)}</li>' for item in summary.get("warnings", [])) or '<li>None.</li>'
    primary = ood_detection.get("overall", {})
    cards = [
        ("Run", summary.get("run_status", "—")),
        ("Evidence", readiness),
        ("Policy-fit rows", summary.get("policy_fit_rows", "—")),
        ("OPE-fit rows", summary.get("ope_fit_rows", "—")),
        ("Seen test", summary.get("seen_test_rows", "—")),
        ("Unseen test", summary.get("unseen_test_rows", "—")),
        ("OOD AUROC", _fmt(primary.get("auroc"), 3)),
        ("OOD AUPRC", _fmt(primary.get("auprc"), 3)),
    ]
    card_html = ''.join(f'<div class="card"><div class="label">{_esc(k)}</div><div class="value">{_esc(v)}</div></div>' for k,v in cards)
    css = """
    :root{--ink:#0f172a;--muted:#64748b;--line:#e2e8f0;--panel:#fff;--bg:#f8fafc;--accent:#2563eb;--ok:#047857;--bad:#b91c1c}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
    main{max-width:1180px;margin:auto;padding:34px 26px 70px}.hero{padding:28px;border:1px solid var(--line);border-radius:18px;background:linear-gradient(135deg,#fff,#eff6ff)}
    h1{font-size:30px;line-height:1.1;margin:0 0 10px}h2{font-size:20px;margin:30px 0 12px}h3{font-size:15px;margin:22px 0 8px}.muted{color:var(--muted)}
    .cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:18px}.card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px}.label{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}.value{font-size:20px;font-weight:750;margin-top:4px;overflow-wrap:anywhere}
    .panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px;margin:12px 0}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:8px 9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#f8fafc;position:sticky;top:0}.gates{list-style:none;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:7px}.gates li{padding:8px 10px;border-radius:8px;background:#f8fafc}.gates span{font-weight:800;margin-right:8px}.gates .ok span{color:var(--ok)}.gates .bad span{color:var(--bad)}
    code{background:#eef2ff;padding:2px 5px;border-radius:5px}.callout{border-left:4px solid #f59e0b;background:#fffbeb;padding:12px 14px;border-radius:8px}.footer{margin-top:35px;color:var(--muted);font-size:11px}@media(max-width:800px){.cards,.grid2,.gates{grid-template-columns:1fr 1fr}}@media(max-width:520px){.cards,.grid2,.gates{grid-template-columns:1fr}}
    svg{width:100%;height:auto;display:block}
    """
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Agentic-RAN publication evidence</title><style>{css}</style></head><body><main>
    <section class="hero"><h1>Agentic-RAN · Full COMMAG Publication Evidence</h1><p class="muted">Self-contained scientific report. Reproduced baselines and counterfactual estimates are kept separate from literature-only PPO results.</p><div class="cards">{card_html}</div></section>
    <h2>How the evaluation works</h2><div class="panel">{_architecture_svg()}<p class="muted">Policy fitting, independent OPE fitting, validation calibration, and final testing use explicitly separated evidence roles. Clustered inference treats each scenario × training configuration × experiment as the primary experimental unit.</p></div>
    <h2>Readiness gates</h2><div class="panel"><ul class="gates">{gate_html}</ul><h3>Warnings / claim restrictions</h3><ul>{warnings}</ul></div>
    <h2>Primary outcomes</h2><div class="grid2"><div class="panel">{_bar_chart(baselines,'selected_ope_mean_utility','Independent-OPE utility')}</div><div class="panel">{_bar_chart(baselines,'selected_ope_sla_violation_rate','Independent-OPE SLA violation',True)}</div><div class="panel">{_bar_chart(baselines,'selected_ope_mean_energy_proxy','Independent-OPE energy proxy',True)}</div><div class="panel">{_bar_chart(baselines,'policy_churn_rate','Policy churn',True)}</div></div>
    <h2>Cluster-level statistical inference</h2><div class="panel">{_forest_plot(clustered_statistics)}<p class="muted">Bootstrap confidence intervals and sign-permutation tests operate on clustered experimental runs, not individual UE traces.</p></div>
    <h2>Distribution shift / OOD</h2><div class="grid2"><div class="panel">{_ood_chart(ood)}</div><div class="panel"><h3>Discrimination metrics</h3>{_table(pd.DataFrame([primary]))}<p class="muted">The OOD threshold is calibrated on seen validation data only; AUROC/AUPRC are evaluated on held-out seen versus unseen conditions.</p></div></div>
    <h2>Validation calibration</h2><div class="panel">{_table(calibration, max_rows=10)}</div>
    <h2>Transition and shortcut audit</h2><div class="grid2"><div class="panel"><h3>Transition audit</h3>{_table(transition_audit, max_rows=12)}</div><div class="panel"><h3>Persistence / shortcut diagnostic</h3>{_table(shortcut, max_rows=12)}</div></div>
    <h2>Latency probes</h2><div class="panel">{_table(pd.DataFrame([latency]))}<p class="muted">Component probes are reported with runtime metadata. They are host/container inference measurements, not end-to-end RIC-to-gNB latency.</p></div>
    <h2>Reproducible baseline table</h2><div class="panel">{_table(baselines, max_rows=20)}</div>
    <h2>Original COMMAG PPO — literature reference only</h2><div class="panel"><p><strong>{_esc(literature_reference.get('paper',{}).get('title',''))}</strong></p><p>{_esc(literature_reference.get('comparison_rule',''))}</p><pre>{_esc(json.dumps(literature_reference.get('reported_results',{}),indent=2))}</pre></div>
    <h2>Scientific interpretation</h2><div class="callout"><strong>Counterfactual guardrail.</strong> Alternative selected-action outcomes are independent direct-method/OPE estimates from fixed logs, not causal online intervention effects. Energy is a normalized proxy rather than measured joules. Results should be presented with clustered confidence intervals and effect sizes, not p-values alone.</div>
    <div class="footer">Generated by <code>agentic_ran.publication_report</code>. No external JavaScript, fonts, images, or CDN resources are required to open this report.</div>
    </main></body></html>"""
    report.write_text(html_doc, encoding="utf-8")
    return report
