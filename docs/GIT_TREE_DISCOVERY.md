# Resilient COMMAG tree discovery

The full publication benchmark must enumerate every UE metrics file at the pinned COMMAG revision. It no longer uses GitHub's recursive REST tree endpoint because transient HTTP 5xx responses can abort an otherwise reproducible study.

The publication workflow now uses Git protocol v2 with a shallow blobless fetch (`--filter=blob:none`) to retrieve only commit/tree metadata for the pinned revision. The resulting path list is cached under `data/raw/commag/.tree-git-<revision>.txt`.

Individual COMMAG CSV files are still handled by the existing downloader and are downloaded only when missing; non-empty cached raw files are reused.
