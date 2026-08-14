# Third-party data

Agentic-RAN source code is MIT-licensed. Public datasets downloaded by its preparation commands retain their own licenses and are not relicensed as MIT.

## Colosseum O-RAN COMMAG

- Repository: <https://github.com/wineslab/colosseum-oran-commag-dataset>
- Pinned revision: `7331cd725fe42b5b9930fdc8acee3006cea00bd7`
- Declared upstream license: GPL-3.0
- Redistribution in this repository: none
- Local generated outputs: `data/raw/commag/` and `data/prepared/commag/`, both ignored by Git

The downloader retrieves selected upstream files at runtime and records their paths and SHA-256 hashes in `commag_manifest.json`. Users who convey upstream data or derived dataset artifacts are responsible for complying with the applicable GPL-3.0 terms and preserving attribution.

## Zenodo measurement sources

The Telenor/COMMECT, Glasgow and optional Vienna source records are listed in `configs/data_sources.yaml`. Their source URLs, checksums and license fields are copied into the generated provenance manifest. Raw files and prepared outputs are ignored by Git.
