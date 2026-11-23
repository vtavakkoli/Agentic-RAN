# Legacy ns-O-RAN Gym / Waf path retired

The previous Docker stage based on the older `wineslab/ns-o-ran-ns3-mmwave`,
`o-ran-e2sim`, `nsoran==0.0.2`, Python 3.11 Waf and GCC12
warning suppression has been **removed from Agentic-RAN's native
Docker runtime**.

**New supported native integration:** [Verified ns-3.49 Gymnasium](DOCKER_MODERN_NS3_GYM.md).

The replacement is built from the independently compiled and
GitHub Actions-verified [vtavakkoli/ns3-gym](https://github.com/vtavakkoli/ns3-gym)
repository at an exact commit, with official ns-3.49 and Python 3.13.
Its native scenario demonstrates actual UDP packet transmissions,
observation/step/reward control, and deterministic fresh episodes.

**Scope:** the new example is **not** NR PHY, O-RAN E2, 5G-LENA,
handover or PRB scheduler. It must never be substituted as a
COMMAG physical research evaluation. For PRB research, provide a
separately validated simulator with correct action and observation
mappings; the publication gate continues to fail until real
compatibility, action receipts, held-out KPIs and baseline superiority
are established.

Previous logs and source references may remain in repository history
for reproducibility; they are not installed into the new native image.
