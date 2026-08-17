# Full COMMAG publication benchmark

This branch extends the compact COMMAG study into a publication-oriented benchmark over multiple RF/mobility scenarios and all 18 published training configurations.

Implementation goals:

- full multi-scenario COMMAG preparation with explicit train/validation/unseen splits
- unseen-scenario OOD evaluation
- classical baselines plus FQI, conservative fitted-Q (CQL-style penalty), and an original-PPO compatibility baseline when upstream model dependencies are available
- policy-shortcut diagnostics that remove current scheduler/PRB features and score only policy-change timestamps
- paired episode-level bootstrap/permutation statistics
- publication report tables that separate observed outcomes from direct-method estimates

The workflow intentionally keeps raw COMMAG files out of this repository and downloads immutable files from the pinned upstream commit.
