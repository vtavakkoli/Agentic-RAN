# Publication preparation recovery

If an earlier publication run failed while calling GitHub's recursive tree API, no dataset reset is required. Pull the current branch and rerun the Compose command. The new preparation path uses Git protocol v2 with a blobless fetch and keeps existing non-empty raw COMMAG files in place.

The previous `.tree-<revision>.json` cache file, if present, is ignored by the new discovery path. The new tree listing is stored as `.tree-git-<revision>.txt` under `data/raw/commag/`.
