"""Evidence-grounded IEEE Access research report for the Agentic-RAN Docker study.

This module never treats a toy Gym reward or native UDP packet delivery as
validated 5G NR scheduler/PRB performance. It intentionally reports negative
and incomplete experiments and prevents stale outputs via run-scoped paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NATIVE_GYM_COMMIT = "21e68565a6beaf13589fd19778b5332f4d0b29e0"
NS3_COMMIT = "418210e8914e9f6bcb5cddbcc94a2cecb7853b68"


def read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("root must be an object")
        return obj, None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"{path.name}: {exc}"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value: Any, digits: int = 5) -> str:
    return f"{value:.{digits}f}" if finite(value) else "not available"


def inspect_native(data: dict[str, Any] | None) -> dict[str, Any]:
    """Cross-check actual receiver values and commands, no fabricated KPIs."""
    out: dict[str, Any] = {"present": data is not None, "passed": False, "errors": [], "episodes": []}
    if data is None:
        out["errors"].append("native real_closed_loop.json not supplied")
        return out
    for field, expected in (
        ("backend", "verified-ns3-3.49-udp-real"),
        ("ns3_sha", NS3_COMMIT),
        ("ns3_gym_sha", NATIVE_GYM_COMMIT),
        ("scenario", "two-node-udp-packets"),
    ):
        if data.get(field) != expected:
            out["errors"].append(f"invalid {field}: {data.get(field)!r}")
    if data.get("commag_prb_controller_evaluated") is not False:
        out["errors"].append("native data must explicitly identify unsupported COMMAG PRB evaluation")
    if data.get("publication_superiority_ready") is not False:
        out["errors"].append("native UDP data must not claim physical RAN publication superiority")
    episodes = data.get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) < 2:
        out["errors"].append("require >=2 real native episodes")
        return out
    for index, episode in enumerate(episodes):
        try:
            seed, trace = episode["seed"], episode["trace"]
            if not isinstance(trace, list) or not trace:
                raise ValueError("missing native step trace")
            old_rx = old_tx = old_bytes = 0
            for step_index, item in enumerate(trace):
                kpi = item["kpis"]
                action = item["applied_action"]
                rx, tx, received_bytes = (int(kpi[x]) for x in ("rx_packets", "tx_packets", "rx_bytes"))
                if (item["step"] != step_index or not isinstance(action, int)
                        or isinstance(action, bool) or not 0 <= action <= 5):
                    raise ValueError("invalid step index or UDP action")
                if (item["requested_action"] != action or kpi.get("applied_action") != action
                        or kpi.get("action_receipt_verified") is not True):
                    raise ValueError("missing or mismatched native action receipt")
                if (tx - old_tx != action or rx < old_rx or rx > tx or received_bytes != 256 * rx
                        or received_bytes < old_bytes):
                    raise ValueError("native transmitter/receiver counters are inconsistent")
                if (not finite(item.get("reward")) or not math.isclose(item["reward"], rx-old_rx, abs_tol=1e-8)):
                    raise ValueError("native reward is not receiver-derived")
                observation = item.get("observation")
                if (not isinstance(observation, list) or len(observation) != 3
                        or [rx, tx, step_index+1] != [int(v) for v in observation]):
                    raise ValueError("observation disagrees with native counters")
                expected_rate = (received_bytes - old_bytes) * 8 / (0.1 * 1e6)
                if (not finite(kpi.get("throughput_mbps")) or
                        not math.isclose(kpi["throughput_mbps"], expected_rate, abs_tol=1e-9)):
                    raise ValueError("native throughput is inconsistent with UDP bytes/step")
                if item.get("truncated") is not False or item.get("terminated") is not (step_index == len(trace)-1):
                    raise ValueError("native termination flags are inconsistent")
                old_rx, old_tx, old_bytes = rx, tx, received_bytes
            final = episode.get("final_kpis", {})
            if any(final.get(k) != last["kpis"].get(k) for last in [trace[-1]] for k in ("rx_packets", "tx_packets", "rx_bytes")):
                raise ValueError("final KPI and last step disagree")
            out["episodes"].append({
                "episode": index+1, "seed": seed, "steps": len(trace), "tx_packets": old_tx,
                "rx_packets": old_rx, "rx_bytes": old_bytes,
                "delivery_fraction": old_rx / old_tx if old_tx else None,
                "mean_throughput_mbps": old_bytes * 8 / (len(trace) * 0.1 * 1e6),
                "total_reward": sum(float(v["reward"]) for v in trace),
            })
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            out["errors"].append(f"episode {index+1}: {exc}")
    if data.get("actual_native_episodes") != len(episodes):
        out["errors"].append("declared episode count does not match evidence")
    if data.get("actual_native_steps") != sum(x.get("steps", 0) for x in out["episodes"]):
        out["errors"].append("declared step count does not match evidence")
    # Check the repeated seed by comparing the actual traces, not merely flags.
    repeated_seed = any(
        e1.get("seed") == e2.get("seed") and e1.get("trace") == e2.get("trace")
        for i, e1 in enumerate(episodes)
        for e2 in episodes[i+1:]
    )
    if not repeated_seed:
        out["errors"].append("no two same-seed episodes have identical traces")
    out["deterministic_replay_observed"] = repeated_seed
    out["passed"] = not out["errors"]
    return out


def inspect_campaign(data: dict[str, Any] | None, root: Path) -> dict[str, Any]:
    out = {"present": data is not None, "passed": False, "errors": [], "comparisons": {}}
    if data is None:
        out["errors"].append("campaign/campaign_summary.json missing")
        return out
    if data.get("backend") != "agentic-ran-interventional-toy-gymnasium":
        out["errors"].append("unexpected campaign backend")
    if data.get("publication_ready_for_ieee_access") is not False or data.get("physical_network_sla_evidence") is not False:
        out["errors"].append("toy campaign falsely claims physical evidence")
    protocol = data.get("protocol", {})
    if not isinstance(protocol, dict):
        protocol = {}
    is_smoke = protocol.get("preset") == "smoke"
    if not is_smoke and (data.get("training_runs", 0) < 8 or data.get("test_seeds_per_run", 0) < 20):
        out["errors"].append("research campaign has too few independent training/test seeds")
    if data.get("total_executed_episodes", 0) <= 0 or data.get("total_executed_steps", 0) <= 0:
        out["errors"].append("no executed Gymnasium episodes")
    comparisons = data.get("comparisons", {})
    if not isinstance(comparisons, dict) or len(comparisons) != 5:
        out["errors"].append("expected five predeclared paired comparisons")
        comparisons = {}
    for name, result in comparisons.items():
        if not isinstance(result, dict) or not finite(result.get("mean_reward_delta")):
            out["errors"].append(f"comparison {name}: invalid reward delta")
            continue
        ci = result.get("train_seed_bootstrap_ci95", [])
        if not isinstance(ci, list) or len(ci) != 2 or not all(finite(v) for v in ci):
            out["errors"].append(f"comparison {name}: invalid cluster bootstrap CI")
        if not finite(result.get("holm_adjusted_p")):
            out["errors"].append(f"comparison {name}: invalid Holm adjusted p")
        out["comparisons"][name] = result
    digest_path = root / "artifact_sha256.json"
    sha_manifest, sha_err = read_json(digest_path)
    if sha_err or sha_manifest is None:
        out["errors"].append("campaign hash manifest missing/invalid")
    else:
        expected_files = {
            "episodes": "all_heldout_episodes.csv.gz", "steps": "all_heldout_steps.csv.gz",
            "run_means": "training_run_means.csv", "scenarios": "scenario_means.csv",
            "summary": "campaign_summary.json", "html": "research_report.html",
        }
        for key, name in expected_files.items():
            file = root / name
            if not file.is_file() or digest(file) != sha_manifest.get(key):
                out["errors"].append(f"missing/mismatched checksum: {name}")
    out.update({
        "smoke": is_smoke, "training_runs": data.get("training_runs"),
        "test_seeds_per_run": data.get("test_seeds_per_run"),
        "executed_episodes": data.get("total_executed_episodes"),
        "executed_steps": data.get("total_executed_steps"),
        "all_positive": data.get("all_comparisons_show_significant_positive_gain") is True,
    })
    out["passed"] = not out["errors"]
    return out


def inspect_offline(data: dict[str, Any] | None) -> dict[str, Any]:
    if data is None:
        return {"present": False, "passed": False, "errors": ["offline/publication_summary.json missing"]}
    gates = data.get("readiness_gates", {})
    if not isinstance(gates, dict):
        gates = {}
    errors = []
    if data.get("run_status") != "EXPERIMENT-COMPLETE":
        errors.append("offline pipeline did not finish")
    if not (data.get("seen_test_rows", 0) > 0 and data.get("unseen_test_rows", 0) > 0):
        errors.append("offline seen/unseen test split not valid")
    if data.get("evidence_status") != "READY-FOR-MANUSCRIPT" or not gates or not all(gates.values()):
        errors.append("internal offline readiness gate failed")
    return {
        "present": True, "passed": not errors, "errors": errors,
        "evidence_status": data.get("evidence_status"),
        "seen_test_rows": data.get("seen_test_rows"), "unseen_test_rows": data.get("unseen_test_rows"),
        "readiness_gates": gates, "warnings": data.get("warnings", []),
        "clustered_statistics": data.get("clustered_statistics", {}),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(rows: list[dict[str, Any]], key: str, value: str, label: str) -> str:
    """Pure SVG, reproducible without a plotting backend or external assets."""
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="70"><text x="12" y="30">No validated data available</text></svg>'
    vals = [float(r[value]) for r in rows]
    bound = max(max(abs(v) for v in vals), 1e-9)
    width = 820
    height = 45 * len(rows) + 65
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(label)}">',
              f'<text x="18" y="23" font-size="16" fill="#15273d">{html.escape(label)}</text>']
    for i, r in enumerate(rows):
        y = 48 + 45 * i
        v = float(r[value]); length = 480 * abs(v) / bound
        fill = "#136f65" if v >= 0 else "#bd5a37"
        pieces.append(f'<text x="18" y="{y+15}" font-size="13">{html.escape(str(r[key]))}</text>')
        pieces.append(f'<rect x="270" y="{y}" width="{length:.2f}" height="21" rx="3" fill="{fill}"/>')
        pieces.append(f'<text x="{285+length:.2f}" y="{y+16}" font-size="12">{v:.5f}</text>')
    pieces.append('</svg>')
    return "".join(pieces)


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p>No verified results available.</p>"
    return "<table><thead><tr>"+"".join(f"<th>{html.escape(c)}</th>" for c in columns)+"</tr></thead><tbody>" + "".join(
        "<tr>"+"".join(f"<td>{html.escape(str(row.get(c, '')))}</td>" for c in columns)+"</tr>" for row in rows
    )+"</tbody></table>"


def produce(run_root: Path, out: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    manifest, manifest_err = read_json(manifest_path or run_root / "run_manifest.json")
    campaign_data, _ = read_json(run_root / "campaign" / "campaign_summary.json")
    offline_data, _ = read_json(run_root / "offline" / "publication_summary.json")
    native_data, _ = read_json(run_root / "native" / "real_closed_loop.json")
    pulse_data, _ = read_json(run_root / "pulse" / "pulse_commag_shadow_summary.json")
    physical_gate, _ = read_json(run_root / "physical" / "final_evidence_gate.json")
    campaign = inspect_campaign(campaign_data, run_root / "campaign")
    native = inspect_native(native_data)
    offline = inspect_offline(offline_data)
    stage_states = manifest.get("stages", {}) if manifest else {}
    flags = manifest.get("options", {}) if manifest else {}
    offline_required = flags.get("include_offline", True)
    native_required = flags.get("include_native", True)
    physical_required = flags.get("include_physical", False)
    physical_pass = bool(isinstance(physical_gate, dict) and physical_gate.get("gate_passed") is True)
    all_stages_success = bool(stage_states) and all(x.get("status") == "passed" for x in stage_states.values())
    package_complete = (campaign["passed"] and (not offline_required or offline["passed"]) and
                        (not native_required or native["passed"]) and all_stages_success and
                        manifest_err is None)
    # No automatic acceptance; physical evidence and independent scholarly review remain necessary.
    physical_prb_validated = physical_pass and physical_required
    checks = {
        "stage_commands_completed": all_stages_success,
        "toy_train_validate_heldout_tests_verified": campaign["passed"],
        "full_offline_commag_required_and_verified": offline["passed"] if offline_required else None,
        "native_udp_episodes_required_and_verified": native["passed"] if native_required else None,
        "real_prb_scheduler_closed_loop_gate": physical_prb_validated,
        "independent_reviewer_and_novelty_assessment": False,
    }
    conclusion = ("DATASET_AND_EXPERIMENT_PACKAGE_COMPLETE; NR_PHYSICAL_GATE_PENDING"
                  if package_complete else "INCOMPLETE_OR_INVALID_EVIDENCE")
    meta = {
        "schema": "agentic-ran-ieee-access-audit-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id") if manifest else None,
        "git_sha": manifest.get("git_sha") if manifest else None,
        "experiment_status": conclusion,
        "publication_ready_for_ieee_access": False,
        "manual_peer_review_required": True,
        "research_package_complete": package_complete,
        "checks": checks,
        "campaign": campaign,
        "offline": offline,
        "native": native,
        "pulse_shadow_present": pulse_data is not None,
        "external_physical_gate_passed": physical_pass,
        "stage_status": stage_states,
        "warnings": ([manifest_err] if manifest_err else []) + [
            "Real ns-3 UDP delivery is not an NR PHY/PRB/scheduler performance experiment.",
            "Campaign reward and SLA proxy arise from an action-dependent toy environment.",
            "COMMAG counterfactual selected-action values are independent direct-method estimates, not observed intervention rewards.",
            "Publication requires external physical-RAN fidelity/actuation evidence, benchmark novelty, literature review, and manual author review.",
        ],
    }
    (out / "ieee_access_evidence.json").write_text(json.dumps(meta, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    comp_rows = []
    for name, row in sorted(campaign["comparisons"].items()):
        ci = row.get("train_seed_bootstrap_ci95", [None, None])
        comp_rows.append({
            "Baseline": name,
            "Mean toy reward delta": number(row.get("mean_reward_delta")),
            "95% cluster CI lower": number(ci[0]) if len(ci) == 2 else "",
            "95% cluster CI upper": number(ci[1]) if len(ci) == 2 else "",
            "Holm-adjusted p": number(row.get("holm_adjusted_p")),
            "Positive significant gain": str(row.get("significant_positive_gain") is True),
            "Toy SLA non-regression": str(row.get("sla_nonregression") is True),
        })
    comp_columns = ["Baseline", "Mean toy reward delta", "95% cluster CI lower", "95% cluster CI upper",
                    "Holm-adjusted p", "Positive significant gain", "Toy SLA non-regression"]
    write_csv(out / "table_toy_comparisons.csv", comp_rows, comp_columns)
    episode_rows = [
        {"Episode": e["episode"], "Seed": e["seed"], "Steps": e["steps"],
         "TX packets": e["tx_packets"], "RX packets": e["rx_packets"], "RX bytes": e["rx_bytes"],
         "Delivery fraction": number(e["delivery_fraction"]),
         "Average Mbps": number(e["mean_throughput_mbps"])} for e in native["episodes"]
    ]
    native_columns = ["Episode", "Seed", "Steps", "TX packets", "RX packets", "RX bytes", "Delivery fraction", "Average Mbps"]
    write_csv(out / "table_native_udp_episodes.csv", episode_rows, native_columns)
    figure_rows = [{"name": r["Baseline"], "delta": float(r["Mean toy reward delta"])} for r in comp_rows]
    svg = bar_chart(figure_rows, "name", "delta", "Paired toy reward delta: PPO-WSP minus baseline")
    (out / "fig_toy_reward_deltas.svg").write_text(svg, encoding="utf-8")
    md = [
        "# IEEE Access – reproducible Agentic-RAN experiment audit",
        "", f"Generated (UTC): {meta['generated_utc']}",
        f"Run: \\`{meta['run_id']}\\`".replace('\\`','`'),
        f"Git commit: \\`{meta['git_sha']}\\`".replace('\\`','`'),
        "", "## Editorial and scientific status", "",
        f"**{conclusion}**", "",
        "**Automatic IEEE Access submission readiness: NO.** The artifact checks evidence completeness; it cannot certify novelty, physical RAN fidelity, or publication acceptance.",
        "", "## Frozen experimental methods", "",
        f"- Independent toy training runs: {campaign.get('training_runs', 'not available')}; held-out test seeds per run: {campaign.get('test_seeds_per_run', 'not available')}.",
        f"- Executed toy episodes: {campaign.get('executed_episodes', 'not available')}; steps: {campaign.get('executed_steps', 'not available')}.",
        "- Six controllers: PPO-WSP, frozen PPO, PULSE-RAN, Twin-CQL, distributional CVaR, and their hybrid; see the frozen campaign protocol.",
        "- PPO-WSP comparisons use *training-run cluster* bootstrap 95% CIs and one-sided train-run sign flips, with Holm multiple-testing correction; report negative outcomes.",
        "- COMMAG is **logged offline evidence** with train/OPE/validation/held-out partitions; alternative-action outcomes are *estimates*, not observed intervention outcomes.",
        "- Native ns-3.49 is a verified **two-node UDP** transport experiment and is a separate experiment from COMMAG PRB scheduling.",
        "", "## Actual native ns-3 network evidence", "",
        f"- Native validation: {'PASS' if native['passed'] else 'NOT VERIFIED'}.",
        f"- Observed native episodes: {len(native['episodes'])}; reproducible same-seed trace: {native.get('deterministic_replay_observed', False)}.",
    ]
    for e in episode_rows:
        md.append(f"- Episode {e['Episode']} (seed {e['Seed']}): {e['TX packets']} TX, {e['RX packets']} RX, {e['RX bytes']} bytes, mean throughput {e['Average Mbps']} Mbps.")
    md += ["", "## Matched toy-policy comparisons", "",
           "| " + " | ".join(comp_columns) + " |",
           "|" + "|".join("---" for _ in comp_columns) + "|"]
    md += ["| " + " | ".join(str(r[x]) for x in comp_columns) + " |" for r in comp_rows]
    md += ["", "## COMMAG offline evidence", "",
           f"- Present: {offline['present']}; internal readiness gates passed: {offline['passed']}.",
           f"- Held-out seen rows: {offline.get('seen_test_rows', 'n/a')}; unseen rows: {offline.get('unseen_test_rows','n/a')}.",
           f"- OPE/cluster statistics: original `offline/` outputs; **no causal superiority claim**.",
           "", "## Evidence status and mandatory reviewer gates", ""]
    for name, state in checks.items():
        md.append(f"- [{'x' if state is True else ' '}] {name}: {state}")
    for kind, part in (("toy campaign", campaign), ("native UDP", native), ("offline COMMAG", offline)):
        for err in part["errors"]:
            md.append(f"- **{kind} issue:** {err}")
    md += ["", "## Suggested IEEE Access manuscript structure", "",
           "1. **Problem & novelty:** formulate safety-governed policy selection with explicit offline-data support limitations; separate novelty from baseline engineering.",
           "2. **System & methods:** architecture; KPI provenance; common action support; calibrated safety/OOD gates; policy/OPE separation; training and testing budgets.",
           "3. **Experimental design:** preregister comparisons, independent training seeds, held-out scenarios, uncertainty and support ablations, baselines, and failure criteria.",
           "4. **Offline COMMAG results:** observed logged outcomes vs independent direct-method estimates, split integrity, confidence intervals, calibration and realism caveats.",
           "5. **Action-dependent toy results:** training-run-cluster paired statistics, corrected p-values, SLA proxies, failure cases, sensitivity and compute cost.",
           "6. **Native ns-3 communication validation:** a separate UDP transport demonstration; do **not** call this NR PRB/slice actuation.",
           "7. **Required new physical experiment:** implement compatible 5G-LENA/NR PRB scheduler action mapping; measure realized per-slice throughput/latency/BLER, BWP capacity and control latency under held-out disturbances.",
           "8. **Threats & reproducibility:** causal/OPE limitations, variation across seeds/environments, source/binary hashes, container versions, data rights, negative results.",
           "", "## IEEE Access submission checklist", "",
           "- Use the official **IEEE Access** double-column single-spaced template, with consistent editable Word/LaTeX and PDF files.",
           "- Provide an original research question and quantitative evidence proportionate to the claims; include current relevant references and conflict/AI-use disclosures as applicable.",
           "- Do not copy this audit as a camera-ready manuscript; figures, references, author biographies and detailed method verification still require author review.",
           "", "Sources: https://ieeeaccess.ieee.org/authors/submission-guidelines/ and https://ieeeaccess.ieee.org/authors/preparing-your-article/.",
           "", "## Files", "",
           "- `ieee_access_evidence.json`: machine-readable validity and provenance gates.",
           "- `table_toy_comparisons.csv`: original signed toy differences and corrected significance.",
           "- `table_native_udp_episodes.csv`: observed native packet counts and measured throughput.",
           "- `fig_toy_reward_deltas.svg`: toy experiment visualization (when evidence exists).",
           "- `sha256_manifest.json`: SHA-256 checksums for produced report artifacts.",
           ""]
    text = "\n".join(md)
    (out / "ieee_access_report.md").write_text(text, encoding="utf-8")
    styled_html = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>IEEE Access – Agentic-RAN evidence audit</title><style>
    body{font:15px/1.55 system-ui,Arial,sans-serif;max-width:1160px;margin:auto;padding:36px;color:#15283b;background:#f7f9fc}
    main{background:#fff;border-radius:18px;padding:34px;box-shadow:0 8px 38px #1f3b5f15}
    h1,h2{color:#123451}h1{font-size:30px}h2{margin-top:30px;border-bottom:1px solid #d6e1ea;padding-bottom:7px}
    .alert{border-left:4px solid #b86a2d;padding:16px;background:#fff5ec}
    .stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.stats div{padding:16px;background:#ecf4f8;border-radius:10px}
    table{border-collapse:collapse;width:100%;font-size:12px;margin:12px 0 18px}th,td{padding:10px;border-bottom:1px solid #dce5ec;text-align:left}
    th{background:#edf3f7}code{background:#f1f4f6;padding:2px 5px;border-radius:4px}svg{width:100%;max-width:900px}
    footer{font-size:12px;color:#5e6c78;margin-top:35px}li{margin:4px 0}</style></head><body><main>'''
    styled_html += f'<h1>Agentic-RAN: reproducible IEEE Access evidence audit</h1><p><strong>Run:</strong> {html.escape(str(meta["run_id"]))} · <strong>Commit:</strong> <code>{html.escape(str(meta["git_sha"]))}</code></p>'
    styled_html += f'<div class="alert"><strong>{html.escape(conclusion)}</strong><p>IEEE Access submission readiness is NOT automatically certified. UDP simulator integration is not NR PRB/scheduler validation.</p></div>'
    styled_html += f'<section class="stats"><div><b>Training runs</b><p>{html.escape(str(campaign.get("training_runs", "n/a")))}</p></div><div><b>Toy held-out episodes</b><p>{html.escape(str(campaign.get("executed_episodes", "n/a")))}</p></div><div><b>Native UDP episodes</b><p>{len(episode_rows)}</p></div></section>'
    styled_html += '<h2>Native ns-3.49 receiver-derived metrics</h2>' + html_table(episode_rows, native_columns)
    styled_html += '<h2>Paired toy Gymnasium policy comparisons</h2>' + html_table(comp_rows, comp_columns) + svg
    styled_html += '<h2>Offline COMMAG audit</h2>' + '<p>'+html.escape(f"Present={offline['present']}; internal checks={offline['passed']}; heldout seen={offline.get('seen_test_rows','n/a')}; unseen={offline.get('unseen_test_rows','n/a')}")+'</p>'
    styled_html += '<h2>Scientific gates</h2><ul>'+''.join(f'<li>{html.escape(name)}: <b>{html.escape(str(state))}</b></li>' for name,state in checks.items())+'</ul>'
    styled_html += '<h2>Paper framing and caveats</h2><p>Read the accompanying Markdown report for the manuscript outline, full limitations, submission checklist and provenance. All offline counterfactual and toy measurements retain their original evidence labels.</p>'
    styled_html += '<footer>Generated by agentic_ran.research_ieee_access_report. Negative outcomes are preserved; missing evidence is not synthesized.</footer></main></body></html>'
    (out / "ieee_access_report.html").write_text(styled_html, encoding="utf-8")
    output_files = sorted(p for p in out.iterdir() if p.is_file() and p.name != "sha256_manifest.json")
    (out / "sha256_manifest.json").write_text(json.dumps({p.name: digest(p) for p in output_files}, indent=2)+"\n", encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit complete train/test/native Docker experiment into an honest research report")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--require-complete", action="store_true", help="Exit 2 if required stages/evidence incomplete; still write report")
    args = ap.parse_args()
    report = produce(args.run_root, args.output, args.manifest)
    print(json.dumps({"experiment_status": report["experiment_status"], "research_package_complete": report["research_package_complete"], "output": str(args.output)}, indent=2), flush=True)
    if args.require_complete and not report["research_package_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()