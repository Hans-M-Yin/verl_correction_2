# Upload selected research data to a private Hugging Face dataset repository

The uploader has an explicit allowlist of 11 parquet files: the four core
training datasets, the Qwen3-VL-8B augmented/control data, and four ratio
ablation files. It does not recurse through `_preprocessed_dataset` and does
not include evaluation results, rollouts, logs, caches, checkpoints, or source
code. The current allowlist totals about 16.5 GiB.

Run the offline preview first:

```bash
python scripts/upload_private_datasets.py --repo-id YOUR_HF_USERNAME/dataset-archive-private
```

Then authenticate and upload:

```bash
hf auth login
python scripts/upload_private_datasets.py \
  --repo-id YOUR_HF_USERNAME/dataset-archive-private \
  --execute
```

The destination must be a private Hugging Face **dataset** repository. The
script creates it as private when absent and refuses an existing public repo
or an existing repo containing unrelated files. It never changes visibility.
An interrupted upload can be rerun with the same command. Files are uploaded
directly from their source locations, so no second 16.5 GiB staging copy is
needed. Initial SHA-256 calculation reads the full parquet files; its cache and
verification record are kept in `/tmp/hf-private-dataset-archive/` by default.

The repository receives the parquet files, `README.md`,
`ARCHIVE_MANIFEST.json`, and `SHA256SUMS`. The manifest intentionally contains
workspace-relative source filenames and descriptions; keep the repository
private. It contains no Hugging Face token. Original parquet files are never
changed or deleted.

The ratio-ablation parquet files exist, but their `stats.json` files are absent
from this workspace, so their intended ratio semantics are marked unverified
in the archive metadata. The archive also records that redistribution terms
for source datasets ViRL-39K and MMK-12 need confirmation before any public
release. This upload is to the requested private repository.

The script uses the installed `huggingface_hub` package. If running on another
machine, install `huggingface_hub>=0.36.2,<1` first.
