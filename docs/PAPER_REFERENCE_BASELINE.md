# COMMAG paper reference baseline

The publication workflow does **not** execute the historical TensorFlow PPO models. The reproducible benchmark runs only the proposed method, FQI, linear CQL, and HistGradientBoosting behavior cloning in the current Python runtime.

For context, the original COMMAG paper is retained as a literature reference:

> L. Bonati, S. D'Oro, M. Polese, S. Basagni, and T. Melodia, "Intelligence and Learning in O-RAN for Data-driven NextG Cellular Networks," IEEE Communications Magazine, vol. 59, no. 10, pp. 21-27, 2021. DOI: 10.1109/MCOM.101.2001120. arXiv:2012.01263.

The paper reports:

- eMBB downlink spectral-efficiency gains **up to 20%** over the best-performing static scheduling policy;
- URLLC average downlink-buffer reductions of **37% vs RR**, **5% vs WF**, and **17% vs PF**.

These values are written to `results/publication/literature_reference.json` by the benchmark.

## Important comparison rule

The paper values are **not reproduced results** from this repository. They use different metrics, action semantics, and experimental/test conditions from the offline direct-method benchmark. Therefore they must not be inserted into `publication_baselines.csv`, used in paired statistical tests, or presented as an apples-to-apples numerical comparison with the proposed method.
